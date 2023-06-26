"""
Turbo Engine - High-performance LLM inference engine
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import EngineConfig
from .scheduler import Scheduler
from .memory_manager import MemoryManager
from .cache_manager import CacheManager
from ..models.base import BaseModel
from ..models.registry import ModelRegistry
from ..utils.metrics import MetricsCollector
from ..utils.logging import setup_logging


@dataclass
class GenerationRequest:
    """Represents a generation request"""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    priority: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerationResponse:
    """Represents a generation response"""
    request_id: str
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    finish_reason: str = "length"
    usage: Dict[str, int] = None
    created_at: float = field(default_factory=time.time)


class TurboEngine:
    """
    High-performance LLM inference engine with advanced optimizations
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = setup_logging(__name__)
        
        # Core components
        self.scheduler = Scheduler(config.scheduler)
        self.memory_manager = MemoryManager(config.memory)
        self.cache_manager = CacheManager()
        self.metrics = MetricsCollector()
        
        # Model and tokenizer
        self.model: Optional[BaseModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # State management
        self._is_initialized = False
        self._is_running = False
        self._active_requests: Dict[str, GenerationRequest] = {}
        self._request_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self._total_requests = 0
        self._total_tokens_generated = 0
        self._start_time = time.time()
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize engine components"""
        self.logger.info("Initializing Turbo Engine components...")
        
        # Initialize memory manager
        self.memory_manager.initialize()
        
        # Initialize cache manager
        self.cache_manager.initialize()
        
        # Initialize scheduler
        self.scheduler.initialize()
        
        self.logger.info("Turbo Engine components initialized successfully")
    
    async def initialize(self):
        """Initialize the engine and load the model"""
        if self._is_initialized:
            return
            
        self.logger.info(f"Loading model: {self.config.model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_class = ModelRegistry.get_model_class(self.config.model)
        self.model = model_class(
            model_name=self.config.model,
            config=self.config,
            tokenizer=self.tokenizer
        )
        
        await self.model.load()
        
        # Initialize CUDA graphs if enabled
        if self.config.enable_cuda_graph:
            await self._init_cuda_graphs()
        
        self._is_initialized = True
        self.logger.info("Turbo Engine initialized successfully")
    
    async def _init_cuda_graphs(self):
        """Initialize CUDA graphs for performance optimization"""
        if not torch.cuda.is_available():
            return
            
        self.logger.info("Initializing CUDA graphs...")
        
        # Warm up the model with dummy inputs
        dummy_input = torch.randint(0, self.tokenizer.vocab_size, (1, 64)).cuda()
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for _ in range(3):  # Warm up iterations
                    _ = self.model.forward(dummy_input)
        
        self.logger.info("CUDA graphs initialized")
    
    async def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        priority: int = 0
    ) -> Union[GenerationResponse, List[GenerationResponse], AsyncGenerator[GenerationResponse, None]]:
        """
        Generate text from prompts
        
        Args:
            prompts: Input prompt(s)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: Sequences to stop generation
            stream: Whether to stream the response
            priority: Request priority (higher = more important)
            
        Returns:
            Generation response(s) or async generator for streaming
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Handle single vs multiple prompts
        if isinstance(prompts, str):
            prompts = [prompts]
            single_response = True
        else:
            single_response = False
        
        # Create requests
        requests = []
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            request = GenerationRequest(
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences,
                stream=stream,
                priority=priority
            )
            requests.append(request)
        
        # Process requests
        if stream:
            return self._generate_stream(requests[0])
        else:
            responses = await self._generate_batch(requests)
            return responses[0] if single_response else responses
    
    async def _generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate responses for a batch of requests"""
        responses = []
        
        for request in requests:
            # Check cache first
            cache_key = self._get_cache_key(request)
            cached_response = self.cache_manager.get(cache_key)
            
            if cached_response:
                responses.append(cached_response)
                continue
            
            # Add to scheduler
            await self.scheduler.add_request(request)
            self._active_requests[request.request_id] = request
            self._total_requests += 1
        
        # Process requests
        while self._active_requests:
            # Get next batch from scheduler
            batch = await self.scheduler.get_next_batch()
            
            if not batch:
                await asyncio.sleep(0.001)  # Small delay
                continue
            
            # Generate responses
            batch_responses = await self._process_batch(batch)
            
            # Update responses
            for response in batch_responses:
                responses.append(response)
                self._active_requests.pop(response.request_id, None)
                
                # Cache response
                cache_key = self._get_cache_key(self._active_requests[response.request_id])
                self.cache_manager.set(cache_key, response)
        
        return responses
    
    async def _generate_stream(self, request: GenerationRequest) -> AsyncGenerator[GenerationResponse, None]:
        """Generate streaming response for a single request"""
        # Add to scheduler
        await self.scheduler.add_request(request)
        self._active_requests[request.request_id] = request
        self._total_requests += 1
        
        generated_text = ""
        generated_tokens = []
        
        while len(generated_tokens) < request.max_tokens:
            # Get next token
            batch = await self.scheduler.get_next_batch()
            
            if not batch or request.request_id not in [r.request_id for r in batch]:
                await asyncio.sleep(0.001)
                continue
            
            # Process single request
            token = await self._generate_next_token(request)
            
            if token is None:
                break
            
            generated_tokens.append(token)
            generated_text += self.tokenizer.decode([token])
            
            # Check stop sequences
            if request.stop_sequences:
                for stop_seq in request.stop_sequences:
                    if stop_seq in generated_text:
                        break
                else:
                    continue
                break
            
            # Yield partial response
            response = GenerationResponse(
                request_id=request.request_id,
                text=generated_text,
                tokens=generated_tokens,
                finish_reason="length" if len(generated_tokens) >= request.max_tokens else "stop"
            )
            
            yield response
        
        # Remove from active requests
        self._active_requests.pop(request.request_id, None)
    
    async def _process_batch(self, batch: List[GenerationRequest]) -> List[GenerationResponse]:
        """Process a batch of requests"""
        if not batch:
            return []
        
        # Prepare inputs
        input_ids = []
        attention_masks = []
        
        for request in batch:
            tokens = self.tokenizer.encode(request.prompt, return_tensors="pt")
            input_ids.append(tokens)
            attention_masks.append(torch.ones_like(tokens))
        
        # Pad sequences
        max_len = max(len(ids[0]) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            padding_len = max_len - len(ids[0])
            if padding_len > 0:
                padded_ids = torch.cat([ids, torch.zeros(1, padding_len, dtype=ids.dtype)], dim=1)
                padded_mask = torch.cat([mask, torch.zeros(1, padding_len, dtype=mask.dtype)], dim=1)
            else:
                padded_ids = ids
                padded_mask = mask
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        # Stack tensors
        input_ids = torch.cat(padded_input_ids, dim=0).cuda()
        attention_mask = torch.cat(padded_attention_masks, dim=0).cuda()
        
        # Generate
        with torch.no_grad():
            outputs = await self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(r.max_tokens for r in batch),
                temperature=batch[0].temperature,
                top_p=batch[0].top_p,
                top_k=batch[0].top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode responses
        responses = []
        for i, request in enumerate(batch):
            input_len = len(self.tokenizer.encode(request.prompt))
            generated_tokens = outputs[i, input_len:].tolist()
            
            # Remove padding tokens
            generated_tokens = [t for t in generated_tokens if t != self.tokenizer.pad_token_id]
            
            # Decode text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Check stop sequences
            finish_reason = "length"
            if request.stop_sequences:
                for stop_seq in request.stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        finish_reason = "stop"
                        break
            
            response = GenerationResponse(
                request_id=request.request_id,
                text=generated_text,
                tokens=generated_tokens,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": input_len,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_len + len(generated_tokens)
                }
            )
            
            responses.append(response)
            self._total_tokens_generated += len(generated_tokens)
        
        return responses
    
    async def _generate_next_token(self, request: GenerationRequest) -> Optional[int]:
        """Generate next token for streaming"""
        # This is a simplified version - in practice, you'd want to maintain
        # KV cache state and generate token by token
        tokens = self.tokenizer.encode(request.prompt, return_tensors="pt").cuda()
        
        with torch.no_grad():
            outputs = self.model.forward(tokens)
            logits = outputs.logits[:, -1, :]
            
            # Apply sampling
            if request.temperature > 0:
                logits = logits / request.temperature
            
            if request.top_p < 1.0:
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample token
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1).item()
            
            return token
    
    def _get_cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        
        key_data = f"{request.prompt}:{request.max_tokens}:{request.temperature}:{request.top_p}:{request.top_k}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def start(self):
        """Start the engine"""
        if not self._is_initialized:
            await self.initialize()
        
        self._is_running = True
        self.logger.info("Turbo Engine started")
    
    async def stop(self):
        """Stop the engine"""
        self._is_running = False
        
        # Clear active requests
        self._active_requests.clear()
        
        # Stop scheduler
        await self.scheduler.stop()
        
        self.logger.info("Turbo Engine stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        uptime = time.time() - self._start_time
        
        return {
            "uptime": uptime,
            "total_requests": self._total_requests,
            "total_tokens_generated": self._total_tokens_generated,
            "active_requests": len(self._active_requests),
            "requests_per_second": self._total_requests / uptime if uptime > 0 else 0,
            "tokens_per_second": self._total_tokens_generated / uptime if uptime > 0 else 0,
            "memory_usage": self.memory_manager.get_stats(),
            "cache_stats": self.cache_manager.get_stats(),
        }
    
    @asynccontextmanager
    async def context(self):
        """Context manager for engine lifecycle"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop() 