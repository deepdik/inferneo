"""
ONNX model implementation for Inferneo
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
import onnxruntime as ort

from ..base import BaseModel, ModelFormat, ModelState, GenerationConfig, GenerationResult


class ONNXModel(BaseModel):
    """
    ONNX model implementation
    
    Supports loading and running ONNX models with optimized inference.
    """
    
    def __init__(self, name: str, config: Any):
        super().__init__(name, config)
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.tokenizer = None  # Will be set during initialization
        
    async def initialize(self, config: Any) -> None:
        """Initialize the ONNX model and load it into memory"""
        self.logger.info(f"Initializing ONNX model: {self.name}")
        self.state = ModelState.LOADING
        
        try:
            # Load ONNX session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(
                self.name,
                providers=providers,
                sess_options=ort.SessionOptions()
            )
            
            # Get input/output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Load tokenizer if provided
            if hasattr(config, 'tokenizer_path'):
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
                
            self.state = ModelState.LOADED
            self.logger.info(f"ONNX model {self.name} initialized successfully")
            
        except Exception as e:
            self.state = ModelState.ERROR
            self.logger.error(f"Failed to initialize ONNX model {self.name}: {e}")
            raise
            
    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """Generate text from a prompt using ONNX model"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not available for ONNX model")
            
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        
        # Prepare input for ONNX
        onnx_inputs = {}
        for i, input_name in enumerate(self.input_names):
            if i < len(inputs):
                onnx_inputs[input_name] = inputs[list(inputs.keys())[i]]
                
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, onnx_inputs)
        
        # Process outputs (this is a simplified version)
        # In practice, you'd implement proper text generation logic
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
        """Generate text for multiple prompts in batch using ONNX model"""
        if not self.is_loaded():
            raise RuntimeError(f"Model {self.name} is not loaded")
            
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not available for ONNX model")
            
        # Use default config if not provided
        if config is None:
            config = GenerationConfig()
            
        # Tokenize inputs
        inputs = self.tokenizer(prompts, return_tensors="np", padding=True, truncation=True)
        
        # Prepare input for ONNX
        onnx_inputs = {}
        for i, input_name in enumerate(self.input_names):
            if i < len(inputs):
                onnx_inputs[input_name] = inputs[list(inputs.keys())[i]]
                
        # Run inference
        start_time = time.time()
        outputs = self.session.run(self.output_names, onnx_inputs)
        
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
        
    async def cleanup(self) -> None:
        """Clean up ONNX model resources"""
        if self.session is not None:
            del self.session
            self.session = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self.state = ModelState.UNLOADED
        self.logger.info(f"Cleaned up ONNX model: {self.name}")
        
    def get_format(self) -> ModelFormat:
        """Get the format of this model"""
        return ModelFormat.ONNX 