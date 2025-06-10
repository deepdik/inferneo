"""
Base model interface for Inferneo

Defines the common interface that all model implementations must follow.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel


class ModelFormat(Enum):
    """Supported model formats"""
    HUGGINGFACE = "huggingface"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TORCHSCRIPT = "torchscript"
    SAFETENSORS = "safetensors"


class ModelState(Enum):
    """Model loading states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    max_length: int = 4096
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    trust_remote_code: bool = False
    revision: Optional[str] = None


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_tokens: Optional[List[str]] = None
    stream: bool = False


@dataclass
class GenerationResult:
    """Result of text generation"""
    text: str
    tokens: List[int]
    finish_reason: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseModel(ABC):
    """
    Abstract base class for all models in Inferneo
    
    This class defines the interface that all model implementations
    must follow, ensuring consistent behavior across different
    model formats and backends.
    """
    
    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.state = ModelState.UNLOADED
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[Any] = None
        
    @abstractmethod
    async def initialize(self, config: Any) -> None:
        """
        Initialize the model and load it into memory
        
        Args:
            config: Configuration object for the model
        """
        pass
        
    @abstractmethod
    async def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> GenerationResult:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            config: Generation configuration (optional)
            
        Returns:
            GenerationResult with generated text and metadata
        """
        pass
        
    @abstractmethod
    async def generate_batch(self, prompts: List[str], config: Optional[GenerationConfig] = None) -> List[GenerationResult]:
        """
        Generate text for multiple prompts in batch
        
        Args:
            prompts: List of input text prompts
            config: Generation configuration (optional)
            
        Returns:
            List of GenerationResult objects
        """
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up model resources and unload from memory
        """
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "format": self.get_format().value,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        
    @abstractmethod
    def get_format(self) -> ModelFormat:
        """
        Get the format of this model
        
        Returns:
            ModelFormat enum value
        """
        pass
        
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.state == ModelState.LOADED
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage of the model
        
        Returns:
            Dictionary with memory usage information
        """
        if not self.is_loaded():
            return {"gpu_memory_mb": 0.0, "cpu_memory_mb": 0.0}
            
        try:
            if torch.cuda.is_available() and hasattr(self.model, 'device'):
                gpu_memory = torch.cuda.memory_allocated(self.model.device) / 1024 / 1024
            else:
                gpu_memory = 0.0
                
            # Estimate CPU memory (this is approximate)
            cpu_memory = 0.0
            if hasattr(self.model, 'num_parameters'):
                cpu_memory = self.model.num_parameters() * 4 / 1024 / 1024  # 4 bytes per parameter
                
            return {
                "gpu_memory_mb": gpu_memory,
                "cpu_memory_mb": cpu_memory
            }
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return {"gpu_memory_mb": 0.0, "cpu_memory_mb": 0.0}
            
    async def health_check(self) -> bool:
        """
        Perform a health check on the model
        
        Returns:
            True if model is healthy, False otherwise
        """
        if not self.is_loaded():
            return False
            
        try:
            # Try to generate a simple response
            result = await self.generate("test", GenerationConfig(max_tokens=1))
            return len(result.text) >= 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


class ModelLoader:
    """Utility class for loading models"""
    
    @staticmethod
    async def load_model(config: ModelConfig) -> BaseModel:
        """
        Load a model based on configuration
        
        Args:
            config: Model configuration
            
        Returns:
            Loaded model instance
        """
        # This is a placeholder - in practice, you'd implement
        # model loading logic based on the model type
        raise NotImplementedError("Model loading not implemented")
    
    @staticmethod
    def get_model_class(model_name: str) -> type:
        """
        Get the appropriate model class for a given model name
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Model class
        """
        # This is a placeholder - in practice, you'd implement
        # logic to determine the appropriate model class
        from .transformers import TransformersModel
        return TransformersModel 