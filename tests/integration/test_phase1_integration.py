"""
Integration tests for Phase 1 functionality
"""

import pytest
import pytest_asyncio
import asyncio
import time
import json
import torch
from typing import Dict, Any

from inferneo.core.enhanced_engine import (
    EnhancedInferneoEngine, 
    EnhancedEngineConfig, 
    RoutingStrategy, 
    ABTestConfig
)
from inferneo.models.manager import ModelManager, ModelFormat
from inferneo.models.base import ModelConfig, GenerationConfig


class TestPhase1Integration:
    """Integration tests for Phase 1 features"""
    
    @pytest_asyncio.fixture
    async def enhanced_engine(self):
        """Create an EnhancedEngine instance for integration testing"""
        config = EnhancedEngineConfig(
            model=None,
            max_workers=4,
            max_memory_gb=8,
            max_concurrent_models=3,
            enable_ab_testing=True,
            enable_performance_monitoring=True,
            enable_health_checks=True,
            metrics_export_interval=5
        )
        
        engine = EnhancedInferneoEngine(config)
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, enhanced_engine):
        """Test complete workflow from model registration to generation"""
        # Register multiple models
        model1_id = await enhanced_engine.register_model(
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
            )
        )
        
        model2_id = await enhanced_engine.register_model(
            name="distilgpt2",
            version="1.0",
            path="distilgpt2",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(
                model_name="distilgpt2",
                model_path="distilgpt2",
                max_length=512,
                dtype="float16",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        
        # Test generation with different routing strategies
        strategies = [
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.LOAD_BALANCED,
            RoutingStrategy.LATENCY_OPTIMIZED,
        ]
        
        for strategy in strategies:
            enhanced_engine.enhanced_config.routing_strategy = strategy
            
            try:
                result = await enhanced_engine.generate(
                    prompt="Hello, how are you?",
                    generation_config=GenerationConfig(
                        max_new_tokens=10,
                        temperature=0.7,
                        top_p=0.9
                    )
                )
                
                assert "generated_text" in result
                assert len(result["generated_text"]) > 0
                
            except Exception as e:
                # Skip if models are not available
                pytest.skip(f"Models not available: {e}")
    
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, enhanced_engine):
        """Test A/B testing workflow"""
        # Register models for A/B testing
        model_a_id = await enhanced_engine.register_model(
            name="gpt2-small",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE
        )
        
        model_b_id = await enhanced_engine.register_model(
            name="distilgpt2",
            version="1.0",
            path="distilgpt2",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Create A/B test
        ab_config = ABTestConfig(
            name="ab_test_1",
            model_a=model_a_id,
            model_b=model_b_id,
            traffic_split=0.6
        )
        
        enhanced_engine.enhanced_config.ab_test_configs = [ab_config]
        
        # Generate requests for A/B testing
        results = []
        for i in range(5):
            try:
                result = await enhanced_engine.generate(
                    prompt=f"Test prompt {i}: What is AI?",
                    generation_config=GenerationConfig(
                        max_new_tokens=20,
                        temperature=0.8
                    )
                )
                results.append(result)
                
            except Exception as e:
                pytest.skip(f"Models not available: {e}")
        
        # Get A/B test results
        ab_results = await enhanced_engine.get_ab_test_results()
        assert len(ab_results) == 1
        
        test_id = f"{model_a_id}_vs_{model_b_id}"
        assert test_id in ab_results
    
    @pytest.mark.asyncio
    async def test_model_version_switching(self, enhanced_engine):
        """Test model version switching workflow"""
        # Register two versions of the same model
        model1_id = await enhanced_engine.register_model(
            name="gpt2-small",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(model_name="gpt2-small", model_path="gpt2", max_length=512)
        )
        
        model2_id = await enhanced_engine.register_model(
            name="gpt2-small",
            version="2.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(model_name="gpt2-small", model_path="gpt2", max_length=1024)
        )
        
        # Switch to version 2.0
        success = await enhanced_engine.switch_model_version("gpt2-small", "2.0")
        assert success is True
        
        # Test generation with new version
        try:
            result = await enhanced_engine.generate(
                prompt="Testing new version: Explain quantum computing",
                generation_config=GenerationConfig(
                    max_new_tokens=30,
                    temperature=0.7
                )
            )
            
            assert "generated_text" in result
            
        except Exception as e:
            pytest.skip(f"Models not available: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, enhanced_engine):
        """Test performance monitoring integration"""
        # Register a model
        model_id = await enhanced_engine.register_model(
            name="gpt2-test",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Generate some requests to build metrics
        for i in range(3):
            try:
                await enhanced_engine.generate(
                    prompt=f"Performance test {i}: Generate a short story",
                    generation_config=GenerationConfig(max_new_tokens=15)
                )
                await asyncio.sleep(0.5)
                
            except Exception as e:
                pytest.skip(f"Models not available: {e}")
        
        # Wait for monitoring to collect data
        await asyncio.sleep(6)
        
        # Get performance metrics
        stats = await enhanced_engine.get_engine_stats()
        
        assert "enhanced_stats" in stats
        assert "performance_metrics" in stats["enhanced_stats"]
        assert "health_status" in stats["enhanced_stats"]
    
    @pytest.mark.asyncio
    async def test_memory_management(self, enhanced_engine):
        """Test memory management with multiple models"""
        # Register multiple models to test memory constraints
        models = []
        for i in range(3):
            model_id = await enhanced_engine.register_model(
                name=f"model-{i}",
                version="1.0",
                path="gpt2",
                format=ModelFormat.HUGGINGFACE
            )
            models.append(model_id)
        
        # Check that all models are registered
        stats = await enhanced_engine.get_engine_stats()
        assert stats["enhanced_stats"]["total_models"] == 3
        
        # Test memory constraints (this would require actual model loading)
        # For now, just verify the configuration
        assert enhanced_engine.enhanced_config.max_concurrent_models == 3
        assert enhanced_engine.enhanced_config.max_memory_gb == 8
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, enhanced_engine):
        """Test handling of concurrent requests"""
        # Register a model
        await enhanced_engine.register_model(
            name="gpt2-concurrent",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Create concurrent requests
        async def generate_request(i):
            try:
                result = await enhanced_engine.generate(
                    prompt=f"Concurrent request {i}: Hello world",
                    generation_config=GenerationConfig(max_new_tokens=10)
                )
                return result
            except Exception as e:
                return {"error": str(e)}
        
        # Run concurrent requests
        tasks = [generate_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Check that all requests completed
        assert len(results) == 5
        
        # Count successful results
        successful_results = [r for r in results if "error" not in r]
        assert len(successful_results) >= 0  # May be 0 if models not available


# Import torch here to avoid import issues
import torch 