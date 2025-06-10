#!/usr/bin/env python3
"""
Model Setup for Production Benchmarks

Downloads and prepares models for both Inferneo and Triton.
"""

import asyncio
import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace any hardcoded Hugging Face token with environment variable
HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN_HERE')  # Set this in your .env file or environment

class ModelSetup:
    """Setup models for benchmarking"""
    
    def __init__(self):
        self.models_dir = Path("./models")
        self.hf_dir = self.models_dir / "huggingface"
        self.triton_dir = self.models_dir / "triton"
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.hf_dir.mkdir(exist_ok=True)
        self.triton_dir.mkdir(exist_ok=True)
        
        # Set environment variable
        os.environ["HF_TOKEN"] = HF_TOKEN
    
    async def download_model(self, model_name: str) -> bool:
        """Download model from HuggingFace"""
        logger.info(f"Downloading model: {model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=HF_TOKEN,
                cache_dir=str(self.hf_dir)
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=HF_TOKEN,
                cache_dir=str(self.hf_dir),
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Save locally
            model_path = self.hf_dir / model_name.replace("/", "_")
            tokenizer.save_pretrained(str(model_path))
            model.save_pretrained(str(model_path))
            
            logger.info(f"Model downloaded to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return False
    
    async def setup_triton_model(self, model_name: str) -> bool:
        """Setup model for Triton server"""
        logger.info(f"Setting up Triton model: {model_name}")
        
        try:
            # Create Triton model directory structure
            triton_model_dir = self.triton_dir / model_name.replace("/", "_")
            triton_model_dir.mkdir(exist_ok=True)
            
            # Create model configuration
            config = {
                "name": model_name.replace("/", "_"),
                "platform": "python",
                "max_batch_size": 32,
                "input": [
                    {
                        "name": "text_input",
                        "data_type": "TYPE_STRING",
                        "dims": [1]
                    }
                ],
                "output": [
                    {
                        "name": "text_output",
                        "data_type": "TYPE_STRING",
                        "dims": [1]
                    }
                ],
                "instance_group": [
                    {
                        "count": 1,
                        "kind": "KIND_GPU"
                    }
                ]
            }
            
            # Save config
            config_path = triton_model_dir / "config.pbtxt"
            with open(config_path, 'w') as f:
                f.write(self._dict_to_config(config))
            
            # Copy model files
            hf_model_path = self.hf_dir / model_name.replace("/", "_")
            if hf_model_path.exists():
                # Copy to Triton format
                triton_model_path = triton_model_dir / "1"
                triton_model_path.mkdir(exist_ok=True)
                
                # Copy model files (simplified - in practice you'd need proper conversion)
                import shutil
                for file in hf_model_path.glob("*.bin"):
                    shutil.copy2(file, triton_model_path)
                for file in hf_model_path.glob("*.safetensors"):
                    shutil.copy2(file, triton_model_path)
                
                logger.info(f"Triton model setup at {triton_model_dir}")
                return True
            else:
                logger.error(f"Source model not found: {hf_model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup Triton model {model_name}: {e}")
            return False
    
    def _dict_to_config(self, config: Dict[str, Any]) -> str:
        """Convert dictionary to Triton config format with quoted strings"""
        def quote(val):
            if isinstance(val, str):
                # Don't quote enum values
                if val in ["TYPE_STRING", "TYPE_INT32", "TYPE_INT64", "TYPE_FP32", "TYPE_FP64", "KIND_GPU", "KIND_CPU"]:
                    return val
                return f'"{val}"'
            return val
        
        lines = []
        for key, value in config.items():
            if isinstance(value, dict):
                lines.append(f"{key} {{")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        lines.append(f"  {sub_key} {{")
                        for item in sub_value:
                            if isinstance(item, dict):
                                lines.append("    {")
                                for item_key, item_value in item.items():
                                    lines.append(f"      {item_key}: {quote(item_value)}")
                                lines.append("    }")
                            else:
                                lines.append(f"    {quote(item)}")
                        lines.append("  }")
                    else:
                        lines.append(f"  {sub_key}: {quote(sub_value)}")
                lines.append("}")
            elif isinstance(value, list):
                # Use square brackets for repeated fields (input, output, instance_group)
                if key in ["input", "output", "instance_group"]:
                    lines.append(f"{key} [")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append("  {")
                            for item_key, item_value in item.items():
                                lines.append(f"    {item_key}: {quote(item_value)}")
                            lines.append("  }")
                        else:
                            lines.append(f"  {quote(item)}")
                    lines.append("]")
                else:
                    lines.append(f"{key} {{")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append("  {")
                            for item_key, item_value in item.items():
                                lines.append(f"    {item_key}: {quote(item_value)}")
                            lines.append("  }")
                        else:
                            lines.append(f"  {quote(item)}")
                    lines.append("}")
            else:
                lines.append(f"{key}: {quote(value)}")
        return "\n".join(lines)
    
    async def setup_all_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Setup all models for benchmarking"""
        results = {}
        
        for model_name in model_names:
            logger.info(f"Setting up {model_name}...")
            
            # Download model
            download_success = await self.download_model(model_name)
            
            # Setup for Triton
            triton_success = await self.setup_triton_model(model_name) if download_success else False
            
            results[model_name] = {
                "download": download_success,
                "triton": triton_success
            }
            
            if download_success and triton_success:
                logger.info(f"✓ {model_name} setup complete")
            else:
                logger.warning(f"✗ {model_name} setup incomplete")
        
        return results
    
    def list_available_models(self) -> List[str]:
        """List available models"""
        models = []
        
        # Check HuggingFace models
        for model_dir in self.hf_dir.iterdir():
            if model_dir.is_dir():
                models.append(model_dir.name.replace("_", "/"))
        
        return models

async def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup models for production benchmarks")
    parser.add_argument("--models", nargs="+", 
                       default=["gpt2", "distilgpt2", "microsoft/DialoGPT-small"],
                       help="Models to setup")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    setup = ModelSetup()
    
    if args.list:
        models = setup.list_available_models()
        print("Available models:")
        for model in models:
            print(f"  - {model}")
    else:
        results = await setup.setup_all_models(args.models)
        
        print("\nSetup Results:")
        for model, result in results.items():
            status = "✓" if result["download"] and result["triton"] else "✗"
            print(f"{status} {model}:")
            print(f"  Download: {'✓' if result['download'] else '✗'}")
            print(f"  Triton: {'✓' if result['triton'] else '✗'}")

if __name__ == "__main__":
    asyncio.run(main()) 