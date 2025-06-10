#!/usr/bin/env python3
"""
Multi-model serving example for Inferneo

Demonstrates how to use the enhanced engine for multi-model serving,
A/B testing, and dynamic routing.
"""

import asyncio
import time
import json

from inferneo.core.enhanced_engine import (
    EnhancedInferneoEngine, 
    EnhancedEngineConfig, 
    RoutingStrategy, 
    ABTestConfig,
    RoutingRule
)
from inferneo.models.manager import ModelFormat
from inferneo.models.base import ModelConfig, GenerationConfig
from inferneo.core.config import EngineConfig


async def multi_model_serving_example():
    """Multi-model serving example"""
    print("🚀 Inferneo - Multi-Model Serving Example")
    print("=" * 60)
    
    # Create enhanced configuration
    config = EngineConfig(
        model="meta-llama/Llama-2-7b-chat-hf",
        max_model_len=4096,
        gpu_memory_utilization=0.9
    )
    
    # Initialize enhanced engine
    engine = EnhancedInferneoEngine(config)
    
    try:
        # Start the engine
        print("Starting Enhanced Inferneo engine...")
        await engine.start()
        
        # Register multiple models
        print("\nRegistering models...")
        
        # Model 1: General purpose
        await engine.register_model(
            name="gpt-3.5-turbo",
            version="1.0",
            path="models/gpt-3.5-turbo",
            format="huggingface",
            config={"max_length": 4096}
        )
        
        # Model 2: Code generation
        await engine.register_model(
            name="code-llama",
            version="1.0", 
            path="models/code-llama",
            format="huggingface",
            config={"max_length": 8192}
        )
        
        # Model 3: Creative writing
        await engine.register_model(
            name="creative-writer",
            version="1.0",
            path="models/creative-writer", 
            format="huggingface",
            config={"max_length": 2048}
        )
        
        # Load models
        print("Loading models...")
        await engine.load_model("gpt-3.5-turbo")
        await engine.load_model("code-llama")
        await engine.load_model("creative-writer")
        
        # Add routing rules
        print("Setting up routing rules...")
        
        # Route code-related prompts to code model
        code_rule = RoutingRule(
            name="code_routing",
            condition="prompt contains 'code' or 'function' or 'class'",
            model_name="code-llama",
            priority=10
        )
        engine.add_routing_rule(code_rule)
        
        # Route creative prompts to creative model
        creative_rule = RoutingRule(
            name="creative_routing", 
            condition="prompt contains 'story' or 'poem' or 'creative'",
            model_name="creative-writer",
            priority=5
        )
        engine.add_routing_rule(creative_rule)
        
        # Add A/B test
        print("Setting up A/B test...")
        ab_test = ABTestConfig(
            name="model_comparison",
            model_a="gpt-3.5-turbo",
            model_b="code-llama",
            traffic_split=0.3  # 30% to model B
        )
        engine.add_ab_test(ab_test)
        
        # Test different types of prompts
        test_prompts = [
            "Write a Python function to sort a list",
            "Tell me a story about a magical forest",
            "What is the capital of France?",
            "Create a class for a bank account",
            "Write a poem about the ocean"
        ]
        
        print("\nTesting different prompts with automatic routing:")
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            result = await engine.generate(prompt, max_tokens=100)
            print(f"Generated: {result.text[:100]}...")
            
        # Test specific model selection
        print("\nTesting specific model selection:")
        result = await engine.generate(
            "Explain quantum computing",
            model_name="gpt-3.5-turbo",
            max_tokens=150
        )
        print(f"GPT-3.5 response: {result.text}")
        
        # Test batch generation
        print("\nTesting batch generation:")
        batch_prompts = [
            "What is machine learning?",
            "Write a function to calculate fibonacci",
            "Tell me about the solar system"
        ]
        
        results = await engine.generate_batch(batch_prompts, max_tokens=80)
        for i, result in enumerate(results):
            print(f"Batch result {i+1}: {result.text[:80]}...")
            
        # Get engine statistics
        stats = engine.get_stats()
        print("\nEnhanced Engine Statistics:")
        print(f"  Loaded models: {stats['loaded_models']}")
        print(f"  Routing rules: {stats['routing_rules']}")
        print(f"  A/B tests: {stats['ab_tests']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        
        # Show model metrics
        print("\nModel Metrics:")
        for model_name, metrics in stats['model_metrics'].items():
            print(f"  {model_name}:")
            print(f"    Requests: {metrics['total_requests']}")
            print(f"    Avg Latency: {metrics['avg_latency']:.3f}s")
            print(f"    Success Rate: {metrics['success_rate']:.2%}")
        
        # Test different routing strategies
        print("\n🛣️ Testing routing strategies...")
        
        routing_strategies = [
            (RoutingStrategy.ROUND_ROBIN, "Round Robin"),
            (RoutingStrategy.LOAD_BALANCED, "Load Balanced"),
            (RoutingStrategy.LATENCY_OPTIMIZED, "Latency Optimized"),
            (RoutingStrategy.QUALITY_OPTIMIZED, "Quality Optimized")
        ]
        
        generation_config = GenerationConfig(
            max_new_tokens=30,
            temperature=0.7,
            top_p=0.9
        )
        
        for strategy, strategy_name in routing_strategies:
            print(f"\n📊 Testing {strategy_name} routing...")
            engine.enhanced_config.routing_strategy = strategy
            
            for i, prompt in enumerate(test_prompts):
                try:
                    start_time = time.time()
                    result = await engine.generate(
                        prompt=prompt,
                        generation_config=generation_config
                    )
                    end_time = time.time()
                    
                    print(f"  Request {i+1}: {end_time - start_time:.3f}s - {len(result.get('generated_text', ''))} chars")
                    
                except Exception as e:
                    print(f"  Request {i+1}: Failed - {e}")
        
        # Test A/B testing
        print("\n🧪 Testing A/B testing...")
        
        # Get A/B test results
        ab_results = await engine.get_ab_test_results()
        print(f"\n📊 A/B test results: {json.dumps(ab_results, indent=2)}")
        
        # Test model version switching
        print("\n🔄 Testing model version switching...")
        
        # Register a new version of the first model
        new_version_id = await engine.register_model(
            name="gpt-3.5-turbo",
            version="2.0",
            path="models/gpt-3.5-turbo",
            format="huggingface",
            config={"max_length": 1024}
        )
        
        print(f"✅ Registered new version: {new_version_id}")
        
        # Switch to new version
        success = await engine.switch_model_version("gpt-3.5-turbo", "2.0")
        print(f"✅ Version switch: {success}")
        
        # Test generation with new version
        try:
            result = await engine.generate(
                prompt="Testing new version: Explain the benefits of renewable energy",
                generation_config=GenerationConfig(
                    max_new_tokens=40,
                    temperature=0.7
                )
            )
            print(f"✅ Generated with new version: {len(result.get('generated_text', ''))} chars")
            
        except Exception as e:
            print(f"❌ Error with new version: {e}")
        
        # Get comprehensive statistics
        print("\n📊 Comprehensive Statistics:")
        stats = await engine.get_engine_stats()
        
        print(f"  Total models: {stats['enhanced_stats']['total_models']}")
        print(f"  Active models: {stats['enhanced_stats']['active_models']}")
        print(f"  Routing strategy: {stats['enhanced_stats']['routing_strategy']}")
        print(f"  A/B tests: {stats['enhanced_stats']['ab_tests']}")
        
        # Performance metrics
        perf_metrics = stats['enhanced_stats']['performance_metrics']
        print(f"  Total requests: {perf_metrics.get('total_requests', 0)}")
        print(f"  Average latency: {perf_metrics.get('avg_latency', 0):.3f}s")
        print(f"  Memory usage: {perf_metrics.get('memory_usage', {}).get('gpu_allocated_gb', 0):.2f} GB")
        
        # Model-specific statistics
        print("\n📊 Model Statistics:")
        for model_id in [new_version_id]:
            model_stats = await engine.get_model_stats(model_id)
            print(f"  {model_id}: {model_stats['state']}, load_count: {model_stats.get('load_count', 0)}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the engine
        print("\nStopping Enhanced Inferneo engine...")
        await engine.stop()
        print("✅ Multi-model serving example completed!")


if __name__ == "__main__":
    # Import torch here to avoid import issues
    import torch
    
    # Run the example
    asyncio.run(multi_model_serving_example()) 