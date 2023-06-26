"""
Base model interface for Turbo Inference Server
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel


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
    """Generation configuration"""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


class BaseModel(ABC):
    """
    Abstract base class for all models in Turbo Inference Server
    """
    
    def __init__(self, model_name: str, config: ModelConfig, tokenizer: AutoTokenizer):
        self.model_name = model_name
        self.config = config
        self.tokenizer = tokenizer
        self.model: Optional[PreTrainedModel] = None
        self._is_loaded = False
        
        # Model metadata
        self.vocab_size: int = 0
        self.hidden_size: int = 0
        self.num_layers: int = 0
        self.num_attention_heads: int = 0
        
        # Performance tracking
        self.total_forward_passes = 0
        self.total_generations = 0
    
    @abstractmethod
    async def load(self) -> None:
        """
        Load the model into memory
        
        This method should be implemented by subclasses to load the specific
        model type (e.g., HuggingFace, custom, etc.)
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from input tokens
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            generation_config: Generation configuration
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        pass
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        Forward pass through the model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_length": self.config.max_length,
            "dtype": str(self.config.dtype),
            "device": self.config.device,
            "is_loaded": self._is_loaded,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            "total_forward_passes": self.total_forward_passes,
            "total_generations": self.total_generations,
            "model_info": self.get_model_info(),
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    async def unload(self) -> None:
        """Unload the model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _extract_model_info(self) -> None:
        """Extract model information from the loaded model"""
        if self.model is None:
            return
        
        # Try to get model info from config
        if hasattr(self.model, 'config'):
            config = self.model.config
            self.vocab_size = getattr(config, 'vocab_size', 0)
            self.hidden_size = getattr(config, 'hidden_size', 0)
            self.num_layers = getattr(config, 'num_hidden_layers', 0)
            self.num_attention_heads = getattr(config, 'num_attention_heads', 0)
        
        # Fallback to tokenizer info
        if self.vocab_size == 0 and self.tokenizer is not None:
            self.vocab_size = self.tokenizer.vocab_size
    
    def _validate_inputs(self, input_ids: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None) -> None:
        """Validate input tensors"""
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be a torch.Tensor")
        
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2-dimensional (batch_size, sequence_length)")
        
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise ValueError("attention_mask must be a torch.Tensor")
            
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids shape")
    
    def _prepare_generation_config(self, 
                                 generation_config: Optional[GenerationConfig] = None) -> GenerationConfig:
        """Prepare generation configuration"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        # Set default token IDs if not provided
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        return generation_config


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