#!/usr/bin/env python3
"""
Triton Python Backend for HuggingFace Models

This script provides a Python backend for Triton to serve HuggingFace models.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

class TritonPythonModel:
    """Python model for Triton serving HuggingFace models"""
    
    def initialize(self, args):
        """Initialize the model"""
        # Model files are in the same directory as this script
        self.model_path = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
            device_map=self.device
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def execute(self, requests):
        """Execute inference requests"""
        responses = []
        
        for request in requests:
            # Get input
            text_input = pb_utils.get_input_tensor_by_name(request, "text_input")
            text = text_input.as_numpy()[0].decode('utf-8')
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("text_output", np.array([generated_text.encode('utf-8')], dtype=np.object_))
            
            # Create response
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
    
    def finalize(self):
        """Clean up resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache() 