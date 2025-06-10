"""
Unit tests for EnhancedEngine
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pytest_asyncio

from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig, RoutingStrategy, ABTestConfig
from inferneo.models.manager import ModelFormat
from inferneo.models.base import ModelConfig, GenerationConfig


class TestEnhancedEngine:
    """Test cases for EnhancedEngine"""
    
    @pytest_asyncio.fixture
    async def enhanced_engine(self):
        """Create an EnhancedEngine instance for testing"""
        config = EnhancedEngineConfig(
            max_workers=2,
            max_memory_gb=4,
            max_concurrent_models=2,
            enable_ab_testing=True,
            enable_performance_monitoring=False,  # Disable for unit tests
            enable_health_checks=False
        )
        config.model = None
        
        engine = EnhancedInferneoEngine(config)
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, enhanced_engine):
        """Test engine initialization"""
        assert enhanced_engine.enhanced_config.max_concurrent_models == 2
        assert enhanced_engine.enhanced_config.enable_ab_testing is True
        assert enhanced_engine.enhanced_config.routing_strategy == RoutingStrategy.LOAD_BALANCED
    
    @pytest.mark.asyncio
    async def test_register_model(self, enhanced_engine):
        """Test model registration in enhanced engine"""
        model_id = await enhanced_engine.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(model_name="test", model_path="test/path")
        )
        
        assert model_id == "test-model:1.0"
        assert model_id in enhanced_engine.model_manager.models
        assert model_id in enhanced_engine.model_loads
        assert enhanced_engine.model_loads[model_id] == 0
    
    @pytest.mark.asyncio
    async def test_routing_strategies(self, enhanced_engine):
        """Test different routing strategies"""
        # Register models
        await enhanced_engine.register_model(
            name="model1",
            version="1.0",
            path="test/path1",
            format=ModelFormat.HUGGINGFACE
        )
        
        await enhanced_engine.register_model(
            name="model2",
            version="1.0",
            path="test/path2",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Test round robin routing
        enhanced_engine.enhanced_config.routing_strategy = RoutingStrategy.ROUND_ROBIN
        model_id1 = await enhanced_engine._route_request()
        model_id2 = await enhanced_engine._route_request()
        
        # Should alternate between models
        assert model_id1 != model_id2
        
        # Test load balanced routing
        enhanced_engine.enhanced_config.routing_strategy = RoutingStrategy.LOAD_BALANCED
        model_id = await enhanced_engine._route_request()
        assert model_id in ["model1:1.0", "model2:1.0"]
    
    @pytest.mark.asyncio
    async def test_ab_test_config(self, enhanced_engine):
        """Test A/B test configuration"""
        # Register models for A/B testing
        model_a_id = await enhanced_engine.register_model(
            name="model-a",
            version="1.0",
            path="test/path-a",
            format=ModelFormat.HUGGINGFACE
        )
        
        model_b_id = await enhanced_engine.register_model(
            name="model-b",
            version="1.0",
            path="test/path-b",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Create A/B test config
        ab_config = ABTestConfig(
            name="ab-test-1",
            model_a=model_a_id,
            model_b=model_b_id,
            traffic_split=0.6
        )
        
        enhanced_engine.enhanced_config.ab_test_configs = [ab_config]
        
        # Test A/B test ID detection
        test_id = enhanced_engine._get_ab_test_id(model_a_id)
        assert test_id == f"{model_a_id}_vs_{model_b_id}"
        
        # Test model selection
        selected_model = await enhanced_engine._select_ab_test_model(test_id)
        assert selected_model in [model_a_id, model_b_id]
    
    @pytest.mark.asyncio
    async def test_switch_model_version(self, enhanced_engine):
        """Test model version switching"""
        # Register two versions
        await enhanced_engine.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        await enhanced_engine.register_model(
            name="test-model",
            version="2.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Switch version
        success = await enhanced_engine.switch_model_version("test-model", "2.0")
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_engine_stats(self, enhanced_engine):
        """Test getting engine statistics"""
        # Register a model
        await enhanced_engine.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        stats = await enhanced_engine.get_engine_stats()
        
        assert "enhanced_stats" in stats
        assert "model_manager_stats" in stats
        assert stats["enhanced_stats"]["total_models"] == 1
        assert stats["enhanced_stats"]["active_models"] == 1
    
    @pytest.mark.asyncio
    async def test_get_model_stats(self, enhanced_engine):
        """Test getting model statistics"""
        model_id = await enhanced_engine.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        stats = await enhanced_engine.get_model_stats(model_id)
        
        assert "name" in stats
        assert "version" in stats
        assert "load_count" in stats
        assert stats["name"] == "test-model"
        assert stats["version"] == "1.0"
        assert stats["load_count"] == 0
    
    @pytest.mark.asyncio
    async def test_get_ab_test_results(self, enhanced_engine):
        """Test getting A/B test results"""
        # Register models
        model_a_id = await enhanced_engine.register_model(
            name="model-a",
            version="1.0",
            path="test/path-a",
            format=ModelFormat.HUGGINGFACE
        )
        
        model_b_id = await enhanced_engine.register_model(
            name="model-b",
            version="1.0",
            path="test/path-b",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Create A/B test
        ab_config = ABTestConfig(
            name="ab-test-1",
            model_a=model_a_id,
            model_b=model_b_id,
            traffic_split=0.5
        )
        
        enhanced_engine.enhanced_config.ab_test_configs = [ab_config]
        
        # Get results
        results = await enhanced_engine.get_ab_test_results()
        assert len(results) == 1
        
        test_id = f"{model_a_id}_vs_{model_b_id}"
        assert test_id in results
    
    @pytest.mark.asyncio
    async def test_update_model_metrics(self, enhanced_engine):
        """Test updating model metrics"""
        model_id = await enhanced_engine.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Update metrics
        await enhanced_engine._update_model_metrics(
            model_id, 
            latency=0.1, 
            result={"quality_score": 0.8}
        )
        
        # Check metrics
        assert len(enhanced_engine.model_latencies[model_id]) == 1
        assert len(enhanced_engine.model_qualities[model_id]) == 1
        assert enhanced_engine.model_latencies[model_id][0] == 0.1
        assert enhanced_engine.model_qualities[model_id][0] == 0.8
    
    @pytest.mark.asyncio
    async def test_calculate_avg_latency(self, enhanced_engine):
        """Test average latency calculation"""
        model_id = "test-model:1.0"
        enhanced_engine.model_latencies[model_id] = [0.1, 0.2, 0.3]
        
        avg_latency = enhanced_engine._calculate_avg_latency(model_id)
        assert avg_latency == 0.2
    
    @pytest.mark.asyncio
    async def test_calculate_avg_quality(self, enhanced_engine):
        """Test average quality calculation"""
        model_id = "test-model:1.0"
        enhanced_engine.model_qualities[model_id] = [0.8, 0.9, 0.7]
        
        avg_quality = enhanced_engine._calculate_avg_quality(model_id)
        assert avg_quality == 0.8 