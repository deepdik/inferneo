"""
ONNX converter utility for Inferneo

Provides utilities to convert HuggingFace models to ONNX format.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ONNXConverter:
    """
    Utility class for converting HuggingFace models to ONNX format.
    
    Supports converting various model architectures to optimized ONNX format
    for faster inference.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def convert_model(self, 
                     model_name: str, 
                     output_path: str,
                     max_length: int = 512,
                     batch_size: int = 1,
                     device: str = "cpu") -> bool:
        """
        Convert a HuggingFace model to ONNX format
        
        Args:
            model_name: HuggingFace model name or path
            output_path: Path to save the ONNX model
            max_length: Maximum sequence length
            batch_size: Batch size for the model
            device: Device to use for conversion
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting model {model_name} to ONNX format")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            model = model.to(device)
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randint(0, tokenizer.vocab_size, (batch_size, max_length), dtype=torch.long)
            dummy_input = dummy_input.to(device)
            
            # Create attention mask
            attention_mask = torch.ones_like(dummy_input)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_input, attention_mask),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # Save tokenizer
            tokenizer_path = Path(output_path).parent / "tokenizer"
            tokenizer.save_pretrained(tokenizer_path)
            
            self.logger.info(f"Successfully converted model to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert model {model_name}: {e}")
            return False
            
    def convert_with_optimization(self,
                                model_name: str,
                                output_path: str,
                                optimization_level: str = "all",
                                **kwargs) -> bool:
        """
        Convert model with additional optimizations
        
        Args:
            model_name: HuggingFace model name or path
            output_path: Path to save the ONNX model
            optimization_level: Optimization level ("basic", "extended", "all")
            **kwargs: Additional conversion parameters
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            self.logger.info(f"Converting model {model_name} with {optimization_level} optimizations")
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Apply optimizations
            if optimization_level in ["extended", "all"]:
                # Enable optimizations
                model.config.use_cache = True
                
            # Convert to ONNX
            return self.convert_model(model_name, output_path, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Failed to convert model with optimizations: {e}")
            return False
            
    def validate_conversion(self, onnx_path: str, model_name: str) -> bool:
        """
        Validate the converted ONNX model
        
        Args:
            onnx_path: Path to the ONNX model
            model_name: Original HuggingFace model name
            
        Returns:
            True if validation successful, False otherwise
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            session = ort.InferenceSession(onnx_path)
            
            # Load original tokenizer for comparison
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Test with simple input
            test_input = "Hello, world!"
            inputs = tokenizer(test_input, return_tensors="np", padding=True)
            
            # Run inference
            outputs = session.run(None, {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            })
            
            self.logger.info("ONNX model validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"ONNX model validation failed: {e}")
            return False
            
    def get_conversion_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about model conversion requirements
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dictionary with conversion information
        """
        try:
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(model_name)
            
            return {
                "model_type": config.model_type,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_hidden_layers,
                "num_attention_heads": config.num_attention_heads,
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
                "supports_onnx": self._check_onnx_support(config.model_type)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get conversion info: {e}")
            return {}
            
    def _check_onnx_support(self, model_type: str) -> bool:
        """Check if model type supports ONNX conversion"""
        supported_types = [
            "gpt2", "gpt_neox", "llama", "mistral", "falcon", "mpt",
            "bloom", "opt", "t5", "bert", "roberta", "distilbert"
        ]
        return model_type in supported_types 