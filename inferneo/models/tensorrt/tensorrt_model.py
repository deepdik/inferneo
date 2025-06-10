"""
TensorRT model implementation for Inferneo
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np

from ..base import BaseModel, ModelFormat, ModelState, GenerationConfig, GenerationResult


class TensorRTModel(BaseModel):
    """
    TensorRT model implementation
    
    Supports loading and running TensorRT optimized models for maximum performance.
    """
    
    def __init__(self, name: str, config: Any):
        super().__init__(name, config)
        self.engine = None
        self.context = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.tokenizer = None
        
    async def initialize(self, config: Any) -> None:
        """Initialize the TensorRT model and load it into memory"""
        self.logger.info(f"Initializing TensorRT model: {self.name}")
        self.state = ModelState.LOADING
        
        try:
            # Import TensorRT
            import tensorrt as trt
            
            # Create TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Load engine from file
            with open(self.name, 'rb') as f:
                engine_data = f.read()
                
            # Create runtime and engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Get input/output names
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)
                    
            # Load tokenizer if provided
            if hasattr(config, 'tokenizer_path'):
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                
            self.state = ModelState.LOADED
            self.logger.info(f"TensorRT model {self.name} initialized successfully")
            
        except Exception as e:
            self.state = ModelState.ERROR
            self.logger.error(f"Failed to initialize TensorRT model {self.name}: {e}")
            raise
            
    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate text from a prompt using TensorRT model"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not available for TensorRT model")
            
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        
        # Prepare input for TensorRT
        trt_inputs = {}
        for i, input_name in enumerate(self.input_names):
            if i < len(inputs):
                trt_inputs[input_name] = inputs[list(inputs.keys())[i]]
                
        # Run inference
        start_time = time.time()
        outputs = await self._run_inference(trt_inputs)
        
        # Process outputs (simplified version)
        logits = outputs[0]  # Assuming first output is logits
        
        # Simple greedy decoding
        generated_tokens = []
        for i in range(min(config.max_tokens, logits.shape[1])):
            next_token = np.argmax(logits[0, i, :])
            generated_tokens.append(int(next_token))
            
        # Decode tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Create result
        result = GenerationResult(
            text=generated_text,
            tokens=generated_tokens,
            finish_reason="length",
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
        
    async def generate_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[GenerationResult]:
        """Generate text for multiple prompts in batch using TensorRT model"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not available for TensorRT model")
            
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
            
        # Tokenize inputs
        inputs = self.tokenizer(prompts, return_tensors="np", padding=True, truncation=True)
        
        # Prepare input for TensorRT
        trt_inputs = {}
        for i, input_name in enumerate(self.input_names):
            if i < len(inputs):
                trt_inputs[input_name] = inputs[list(inputs.keys())[i]]
                
        # Run inference
        start_time = time.time()
        outputs = await self._run_inference(trt_inputs)
        
        # Process outputs
        logits = outputs[0]
        results = []
        
        for batch_idx in range(logits.shape[0]):
            generated_tokens = []
            for i in range(min(config.max_tokens, logits.shape[1])):
                next_token = np.argmax(logits[batch_idx, i, :])
                generated_tokens.append(int(next_token))
                
            # Decode tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Create result
            result = GenerationResult(
                text=generated_text,
                tokens=generated_tokens,
                finish_reason="length",
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
            
            results.append(result)
            
        return results
        
    async def _run_inference(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference with TensorRT engine"""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Allocate GPU memory
        gpu_inputs = {}
        gpu_outputs = {}
        cpu_outputs = {}
        
        for name, data in inputs.items():
            gpu_inputs[name] = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(gpu_inputs[name], data)
            
        # Allocate output memory
        for name in self.output_names:
            size = self.engine.get_binding_size(self.engine.get_binding_index(name))
            gpu_outputs[name] = cuda.mem_alloc(size)
            cpu_outputs[name] = np.empty(size // 4, dtype=np.float32)  # Assuming float32
            
        # Create bindings
        bindings = []
        for name in self.input_names:
            bindings.append(int(gpu_inputs[name]))
        for name in self.output_names:
            bindings.append(int(gpu_outputs[name]))
            
        # Run inference
        self.context.execute_v2(bindings)
        
        # Copy outputs back to CPU
        outputs = []
        for name in self.output_names:
            cuda.memcpy_dtoh(cpu_outputs[name], gpu_outputs[name])
            outputs.append(cpu_outputs[name])
            
        return outputs
        
    async def cleanup(self) -> None:
        """Clean up TensorRT model resources"""
        if self.context is not None:
            del self.context
            self.context = None
            
        if self.engine is not None:
            del self.engine
            self.engine = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self.state = ModelState.UNLOADED
        self.logger.info(f"Cleaned up TensorRT model: {self.name}")
        
    def get_format(self) -> ModelFormat:
        """Get the format of this model"""
        return ModelFormat.TENSORRT 