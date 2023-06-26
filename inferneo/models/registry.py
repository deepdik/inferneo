"""
Model registry for managing different model types and implementations
"""

import re
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass

from .base import BaseModel


@dataclass
class ModelInfo:
    """Information about a registered model"""
    name: str
    model_class: Type[BaseModel]
    supported_formats: List[str]
    description: str
    default_config: Dict[str, Any]


class ModelRegistry:
    """
    Registry for managing different model types and implementations
    """
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._patterns: List[tuple] = []  # (pattern, model_class)
        self._default_model: Optional[Type[BaseModel]] = None
    
    def register_model(self, 
                      name: str,
                      model_class: Type[BaseModel],
                      supported_formats: List[str] = None,
                      description: str = "",
                      default_config: Dict[str, Any] = None,
                      pattern: Optional[str] = None) -> None:
        """
        Register a model class
        
        Args:
            name: Model name
            model_class: Model class to register
            supported_formats: Supported model formats
            description: Model description
            default_config: Default configuration
            pattern: Regex pattern for matching model names
        """
        if supported_formats is None:
            supported_formats = ["*"]
        
        if default_config is None:
            default_config = {}
        
        # Register by name
        self._models[name] = ModelInfo(
            name=name,
            model_class=model_class,
            supported_formats=supported_formats,
            description=description,
            default_config=default_config
        )
        
        # Register pattern if provided
        if pattern:
            compiled_pattern = re.compile(pattern)
            self._patterns.append((compiled_pattern, model_class))
    
    def get_model_class(self, model_name: str) -> Type[BaseModel]:
        """
        Get model class for a given model name
        
        Args:
            model_name: Name or path of the model
            
        Returns:
            Model class
        """
        # First, try exact match
        if model_name in self._models:
            return self._models[model_name].model_class
        
        # Then, try pattern matching
        for pattern, model_class in self._patterns:
            if pattern.match(model_name):
                return model_class
        
        # Finally, return default model
        if self._default_model:
            return self._default_model
        
        raise ValueError(f"No model class found for: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get model information
        
        Args:
            model_name: Model name
            
        Returns:
            Model information or None if not found
        """
        return self._models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self._models.keys())
    
    def list_model_info(self) -> List[ModelInfo]:
        """List all registered model information"""
        return list(self._models.values())
    
    def set_default_model(self, model_class: Type[BaseModel]) -> None:
        """Set default model class"""
        self._default_model = model_class
    
    def unregister_model(self, name: str) -> bool:
        """
        Unregister a model
        
        Args:
            name: Model name to unregister
            
        Returns:
            True if model was unregistered
        """
        if name in self._models:
            del self._models[name]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered models"""
        self._models.clear()
        self._patterns.clear()
        self._default_model = None
    
    def is_supported(self, model_name: str) -> bool:
        """
        Check if a model is supported
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is supported
        """
        try:
            self.get_model_class(model_name)
            return True
        except ValueError:
            return False
    
    def get_supported_formats(self, model_name: str) -> List[str]:
        """
        Get supported formats for a model
        
        Args:
            model_name: Model name
            
        Returns:
            List of supported formats
        """
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.supported_formats
        return []
    
    def get_default_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a model
        
        Args:
            model_name: Model name
            
        Returns:
            Default configuration
        """
        model_info = self.get_model_info(model_name)
        if model_info:
            return model_info.default_config.copy()
        return {}


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry"""
    return _registry


def register_model(name: str,
                  model_class: Type[BaseModel],
                  supported_formats: List[str] = None,
                  description: str = "",
                  default_config: Dict[str, Any] = None,
                  pattern: Optional[str] = None) -> None:
    """Register a model with the global registry"""
    _registry.register_model(
        name=name,
        model_class=model_class,
        supported_formats=supported_formats,
        description=description,
        default_config=default_config,
        pattern=pattern
    )


def get_model_class(model_name: str) -> Type[BaseModel]:
    """Get model class from global registry"""
    return _registry.get_model_class(model_name)


def is_supported(model_name: str) -> bool:
    """Check if model is supported in global registry"""
    return _registry.is_supported(model_name)


# Auto-register common model types
def _auto_register_models():
    """Auto-register common model types"""
    try:
        from .transformers import TransformersModel
        register_model(
            name="transformers",
            model_class=TransformersModel,
            supported_formats=["huggingface", "safetensors", "pytorch"],
            description="HuggingFace Transformers models",
            default_config={
                "trust_remote_code": False,
                "revision": None,
                "dtype": "float16"
            },
            pattern=r".*"
        )
        _registry.set_default_model(TransformersModel)
    except ImportError:
        pass
    
    try:
        from .custom import CustomModel
        register_model(
            name="custom",
            model_class=CustomModel,
            supported_formats=["custom"],
            description="Custom model implementations",
            default_config={}
        )
    except ImportError:
        pass


# Initialize auto-registration
_auto_register_models() 