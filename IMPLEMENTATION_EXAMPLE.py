#!/usr/bin/env python3
"""
Practical Implementation Example for Inferneo
Demonstrates all requested features: lazy loading, multi-model serving, versioning, etc.
"""

import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass

# Import our enhanced components
from inferneo.core.enhanced_engine import EnhancedInferneoEngine
from inferneo.core.config import EngineConfig
from inferneo.models.manager import ModelManager, ModelFormat
from inferneo.models.base import ModelConfig


@dataclass
class ModelSpec:
    """Model specification for registration"""
    name: str
    version: str
    path: str
    format: ModelFormat
    config: Dict[str, Any]
    metadata: Dict[str, Any]


class InferneoServer:
    """
    Complete Inferneo Server with all requested features
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.engine = EnhancedInferneoEngine(config)
        self.model_manager = self.engine.model_manager
        
        # Model specifications
        self.registered_models: Dict[str, ModelSpec] = {}
        
        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
    
    async def start(self):
        """Start the server"""
        await self.engine.start()
        print("🚀 Inferneo Server started")
    
    async def stop(self):
        """Stop the server"""
        await self.engine.stop()
        print("🛑 Inferneo Server stopped")
    
    async def register_model(self, spec: ModelSpec) -> str:
        """
        Register a model for lazy loading
        
        Args:
            spec: Model specification
            
        Returns:
            Model ID
        """
        model_id = await self.model_manager.register_model(
            name=spec.name,
            version=spec.version,
            path=spec.path,
            format=spec.format,
            config=spec.config,
            metadata=spec.metadata
        )
        
        self.registered_models[model_id] = spec
        print(f"✅ Registered model: {model_id}")
        
        return model_id
    
    async def generate_text(self, 
                          prompt: str,
                          model_name: Optional[str] = None,
                          model_version: Optional[str] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate text with automatic model routing
        
        Args:
            prompt: Input prompt
            model_name: Specific model to use
            model_version: Specific model version
            **kwargs: Generation parameters
            
        Returns:
            Generation response
        """
        self.request_count += 1
        
        # Route to appropriate model
        routing_rules = {}
        if model_name:
            routing_rules["model"] = model_name
        if model_version:
            routing_rules["version"] = model_version
        
        # Generate response
        response = await self.engine.generate(
            prompts=prompt,
            routing_rules=routing_rules,
            **kwargs
        )
        
        return {
            "text": response.text,
            "tokens": response.tokens,
            "model_name": response.model_name,
            "model_version": response.model_version,
            "processing_time": response.processing_time,
            "usage": response.usage
        }
    
    async def switch_model_version(self, model_name: str, version: str) -> bool:
        """Switch to a different model version"""
        success = await self.engine.switch_model_version(model_name, version)
        if success:
            print(f"✅ Switched {model_name} to version {version}")
        else:
            print(f"❌ Failed to switch {model_name} to version {version}")
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics"""
        engine_stats = self.engine.get_stats()
        manager_stats = await self.engine.get_manager_stats()
        
        uptime = time.time() - self.start_time
        
        return {
            "server": {
                "uptime": uptime,
                "total_requests": self.request_count,
                "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            },
            "engine": engine_stats,
            "model_manager": manager_stats,
            "registered_models": list(self.registered_models.keys())
        }


# Example usage
async def main():
    """Example usage of the enhanced Inferneo Server"""
    
    # Configuration
    config = EngineConfig(
        model="meta-llama/Llama-2-7b-chat-hf",  # Default model
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        enable_cuda_graph=True,
        enable_flash_attention=True,
        enable_xformers=True
    )
    
    # Create server
    server = InferneoServer(config)
    
    try:
        # Start server
        await server.start()
        
        # Register multiple models (lazy loading)
        models = [
            ModelSpec(
                name="llama-7b",
                version="v1.0",
                path="meta-llama/Llama-2-7b-chat-hf",
                format=ModelFormat.HUGGINGFACE,
                config={
                    "max_length": 4096,
                    "dtype": "float16",
                    "use_flash_attention": True
                },
                metadata={"description": "Llama 2 7B Chat model"}
            ),
            ModelSpec(
                name="llama-13b",
                version="v1.0",
                path="meta-llama/Llama-2-13b-chat-hf",
                format=ModelFormat.HUGGINGFACE,
                config={
                    "max_length": 4096,
                    "dtype": "float16",
                    "use_flash_attention": True
                },
                metadata={"description": "Llama 2 13B Chat model"}
            ),
            ModelSpec(
                name="llama-7b",
                version="v2.0",
                path="meta-llama/Llama-2-7b-chat-hf",
                format=ModelFormat.HUGGINGFACE,
                config={
                    "max_length": 8192,  # Extended context
                    "dtype": "float16",
                    "use_flash_attention": True
                },
                metadata={"description": "Llama 2 7B with extended context"}
            )
        ]
        
        # Register models
        for model_spec in models:
            await server.register_model(model_spec)
        
        print("\n🧪 Testing Multi-Model Generation...")
        
        # Test 1: Basic generation (auto-routing)
        response1 = await server.generate_text(
            prompt="Hello, how are you?",
            max_tokens=50,
            temperature=0.7
        )
        print(f"✅ Auto-routed response: {response1['text'][:100]}...")
        print(f"   Model: {response1['model_name']} v{response1['model_version']}")
        print(f"   Time: {response1['processing_time']:.3f}s")
        
        # Test 2: Specific model generation
        response2 = await server.generate_text(
            prompt="Explain quantum computing in simple terms.",
            model_name="llama-13b",
            max_tokens=100,
            temperature=0.8
        )
        print(f"✅ Specific model response: {response2['text'][:100]}...")
        print(f"   Model: {response2['model_name']} v{response2['model_version']}")
        print(f"   Time: {response2['processing_time']:.3f}s")
        
        # Test 3: Model version switching
        print("\n🔄 Testing Model Version Switching...")
        await server.switch_model_version("llama-7b", "v2.0")
        
        response3 = await server.generate_text(
            prompt="Write a long story about a magical forest.",
            model_name="llama-7b",
            max_tokens=200,
            temperature=0.9
        )
        print(f"✅ Version-switched response: {response3['text'][:100]}...")
        print(f"   Model: {response3['model_name']} v{response3['model_version']}")
        print(f"   Time: {response3['processing_time']:.3f}s")
        
        # Test 4: Performance testing
        print("\n⚡ Performance Testing...")
        start_time = time.time()
        
        # Generate multiple requests concurrently
        tasks = []
        for i in range(10):
            task = server.generate_text(
                prompt=f"Generate response {i+1}: What is AI?",
                max_tokens=50,
                temperature=0.7
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        print(f"✅ Generated {len(responses)} responses in {total_time:.3f}s")
        print(f"   Average time per request: {total_time/len(responses):.3f}s")
        print(f"   Throughput: {len(responses)/total_time:.1f} requests/sec")
        
        # Get comprehensive stats
        stats = await server.get_stats()
        print("\n📊 Server Statistics:")
        print(f"   Total requests: {stats['server']['total_requests']}")
        print(f"   Requests/sec: {stats['server']['requests_per_second']:.1f}")
        print(f"   Loaded models: {stats['model_manager']['loaded_models']}")
        print(f"   Active models: {stats['model_manager']['active_models']}")
        print(f"   Memory usage: {stats['model_manager']['total_memory_gb']:.2f} GB")
        print(f"   Cache hit rate: {stats['model_manager']['hit_rate']:.2%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop server
        await server.stop()


# Advanced features demonstration
async def demonstrate_advanced_features():
    """Demonstrate advanced features like A/B testing and dynamic routing"""
    
    config = EngineConfig(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_model_len=4096,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9
    )
    
    server = InferneoServer(config)
    
    try:
        await server.start()
        
        # Register models for A/B testing
        await server.register_model(ModelSpec(
            name="experimental-model",
            version="control",
            path="meta-llama/Llama-2-7b-chat-hf",
            format=ModelFormat.HUGGINGFACE,
            config={"max_length": 4096, "dtype": "float16"},
            metadata={"ab_test_group": "control"}
        ))
        
        await server.register_model(ModelSpec(
            name="experimental-model",
            version="treatment",
            path="meta-llama/Llama-2-7b-chat-hf",
            format=ModelFormat.HUGGINGFACE,
            config={"max_length": 4096, "dtype": "float16", "use_flash_attention": True},
            metadata={"ab_test_group": "treatment"}
        ))
        
        print("\n🧪 A/B Testing Demonstration...")
        
        # Simulate A/B testing
        for i in range(20):
            # Route 50% to control, 50% to treatment
            version = "control" if i % 2 == 0 else "treatment"
            
            response = await server.generate_text(
                prompt=f"Test prompt {i+1}: Explain machine learning.",
                model_name="experimental-model",
                model_version=version,
                max_tokens=50,
                temperature=0.7
            )
            
            print(f"Request {i+1} -> {version}: {response['processing_time']:.3f}s")
        
        # Get A/B testing stats
        stats = await server.get_stats()
        print(f"\n📊 A/B Testing Results:")
        print(f"   Total requests: {stats['server']['total_requests']}")
        print(f"   Average response time: {stats['engine']['avg_response_time']:.3f}s")
        
    finally:
        await server.stop()


if __name__ == "__main__":
    print("🚀 Inferneo - Advanced Features Demo")
    print("=" * 60)
    
    # Run basic demo
    asyncio.run(main())
    
    print("\n" + "=" * 60)
    print("🧪 Advanced Features Demo")
    print("=" * 60)
    
    # Run advanced features demo
    asyncio.run(demonstrate_advanced_features())
    
    print("\n✅ Demo completed successfully!")
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Lazy loading of models")
    print("   ✅ Multi-model serving")
    print("   ✅ Dynamic request routing")
    print("   ✅ Model versioning and switching")
    print("   ✅ A/B testing capability")
    print("   ✅ Performance monitoring")
    print("   ✅ Memory management")
    print("   ✅ Concurrent request handling") 