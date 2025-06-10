"""
HuggingFace Transformers model implementation for Inferneo
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from .base import BaseModel, ModelFormat, ModelState, GenerationConfig as BaseGenerationConfig, GenerationResult


class TransformersModel(BaseModel):
    """
    HuggingFace Transformers model implementation
    
    Supports loading and running HuggingFace models with advanced
    optimizations and memory management.
    """
    
    def __init__(self, name: str, config: Any):
        super().__init__(name, config)
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def initialize(self, config: Any) -> None:
        """Initialize the model and load it into memory"""
        self.logger.info(f"Initializing Transformers model: {self.name}")
        self.state = ModelState.LOADING
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.name,
                trust_remote_code=getattr(config, 'trust_remote_code', True),
                revision=getattr(config, 'revision', None)
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=getattr(config, 'trust_remote_code', True),
                revision=getattr(config, 'revision', None),
                low_cpu_mem_usage=True
            )
            
            # Move to device if not using device_map
            if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
                self.model = self.model.to(self.device)
                
            # Set to evaluation mode
            self.model.eval()
            
            self.state = ModelState.LOADED
            self.logger.info(f"Transformers model {self.name} initialized successfully")
            
        except Exception as e:
            self.state = ModelState.ERROR
            self.logger.error(f"Failed to initialize model {self.name}: {e}")
            raise
            
    async def generate(self, prompt: str, config: Optional[BaseGenerationConfig] = None) -> GenerationResult:
        """Generate text from a prompt"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        # Use default config if not provided
        if config is None:
            config = BaseGenerationConfig()
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False
            )
            
        # Decode output
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:].tolist()
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Determine finish reason
        finish_reason = "length"
        if len(generated_tokens) < config.max_tokens:
            finish_reason = "stop"
            
        # Create result
        result = GenerationResult(
            text=generated_text,
            tokens=generated_tokens,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": len(generated_tokens),
                "total_tokens": inputs['input_ids'].shape[1] + len(generated_tokens)
            },
            metadata={
                "model": self.name,
                "generation_time": time.time() - start_time
            }
        )
        
        return result
        
    async def generate_batch(self, prompts: List[str], config: Optional[BaseGenerationConfig] = None) -> List[GenerationResult]:
        """Generate text for multiple prompts in batch"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        # Use default config if not provided
        if config is None:
            config = BaseGenerationConfig()
            
        # Tokenize inputs
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        
        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False
            )
            
        # Process results
        results = []
        for i, sequence in enumerate(outputs.sequences):
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = sequence[input_length:].tolist()
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Determine finish reason
            finish_reason = "length"
            if len(generated_tokens) < config.max_tokens:
                finish_reason = "stop"
                
            # Create result
            result = GenerationResult(
                text=generated_text,
                tokens=generated_tokens,
                finish_reason=finish_reason,
                usage={
                    "prompt_tokens": input_length,
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": input_length + len(generated_tokens)
                },
                metadata={
                    "model": self.name,
                    "generation_time": time.time() - start_time
                }
            )
            
            results.append(result)
            
        return results
        
    async def cleanup(self) -> None:
        """Clean up model resources and unload from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.state = ModelState.UNLOADED
        self.logger.info(f"Cleaned up model: {self.name}")
        
    def get_format(self) -> ModelFormat:
        """Get the format of this model"""
        return ModelFormat.HUGGINGFACE 