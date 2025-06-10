#!/usr/bin/env python3
"""
Basic usage example for Inferneo

Demonstrates how to use the core engine for text generation.
"""

import asyncio
import time

from inferneo.core.enhanced_engine import EnhancedEngine, EnhancedEngineConfig
from inferneo.models.manager import ModelFormat
from inferneo.models.base import ModelConfig, GenerationConfig


async def basic_usage_example():
    """Basic usage example"""
    print("🚀 Inferneo - Basic Usage Example")
    print("=" * 50)
    
    # Create enhanced engine configuration
    config = EnhancedEngineConfig(
        max_workers=4,
        max_memory_gb=8,
        max_concurrent_models=2,
        enable_ab_testing=True,
        enable_performance_monitoring=True,
        enable_health_checks=True
    )
    
    # Create enhanced engine
    engine = EnhancedEngine(config)
    await engine.start()
    
    try:
        print("📝 Registering models...")
        
        # Register a model
        model_id = await engine.register_model(
            name="gpt2-small",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(
                model_name="gpt2",
                model_path="gpt2",
                max_length=512,
                dtype="float16",
                device="cuda" if torch.cuda.is_available() else "cpu"
            ),
            metadata={"description": "Small GPT-2 model for basic example"}
        )
        
        print(f"✅ Registered model: {model_id}")
        
        # Generate text
        print("\n🤖 Generating text...")
        
        prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Write a short story about a robot.",
            "Explain quantum computing in simple terms."
        ]
        
        generation_config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt {i}: {prompt}")
            
            try:
                start_time = time.time()
                result = await engine.generate(
                    prompt=prompt,
                    generation_config=generation_config
                )
                end_time = time.time()
                
                print(f"⏱️  Generation time: {end_time - start_time:.3f}s")
                print(f"🤖 Generated: {result['generated_text']}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")
        
        # Get engine statistics
        print("\n📊 Engine Statistics:")
        stats = await engine.get_engine_stats()
        
        print(f"  Total models: {stats['enhanced_stats']['total_models']}")
        print(f"  Active models: {stats['enhanced_stats']['active_models']}")
        print(f"  Total requests: {stats['enhanced_stats']['performance_metrics'].get('total_requests', 0)}")
        
        # Get model statistics
        print("\n📊 Model Statistics:")
        model_stats = await engine.get_model_stats(model_id)
        
        print(f"  Model: {model_stats['name']}:{model_stats['version']}")
        print(f"  State: {model_stats['state']}")
        print(f"  Load count: {model_stats.get('load_count', 0)}")
        print(f"  Memory usage: {model_stats.get('memory_usage', 0)} bytes")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await engine.stop()
        print("\n✅ Basic usage example completed!")


if __name__ == "__main__":
    # Import torch here to avoid import issues
    import torch
    
    # Run the example
    asyncio.run(basic_usage_example()) 