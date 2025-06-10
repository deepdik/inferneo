"""
Unit tests for ModelManager
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pytest_asyncio

from inferneo.models.manager import ModelManager, ModelFormat, ModelState
from inferneo.models.base import ModelConfig


class TestModelManager:
    """Test cases for ModelManager"""
    
    @pytest_asyncio.fixture
    async def model_manager(self):
        """Create a ModelManager instance for testing"""
        manager = ModelManager(
            max_models=3,
            max_memory_gb=8,
            enable_lazy_loading=True,
            enable_model_caching=True
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_register_model(self, model_manager):
        """Test model registration"""
        model_id = await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE,
            config=ModelConfig(model_name="test", model_path="test/path")
        )
        
        assert model_id == "test-model:1.0"
        assert model_id in model_manager.models
        assert model_manager.models[model_id].name == "test-model"
        assert model_manager.models[model_id].version == "1.0"
        assert model_manager.models[model_id].format == ModelFormat.HUGGINGFACE
    
    @pytest.mark.asyncio
    async def test_register_duplicate_model(self, model_manager):
        """Test registering duplicate model raises error"""
        await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        with pytest.raises(ValueError, match="already registered"):
            await model_manager.register_model(
                name="test-model",
                version="1.0",
                path="test/path",
                format=ModelFormat.HUGGINGFACE
            )
    
    @pytest.mark.asyncio
    async def test_model_versioning(self, model_manager):
        """Test model versioning functionality"""
        # Register first version
        model1_id = await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Register second version
        model2_id = await model_manager.register_model(
            name="test-model",
            version="2.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        assert model1_id == "test-model:1.0"
        assert model2_id == "test-model:2.0"
        assert "test-model" in model_manager.versions
        assert len(model_manager.versions["test-model"]) == 2
    
    @pytest.mark.asyncio
    async def test_switch_version(self, model_manager):
        """Test model version switching"""
        # Register two versions
        await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        await model_manager.register_model(
            name="test-model",
            version="2.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Switch to version 2.0
        success = await model_manager.switch_version("test-model", "2.0")
        assert success is True
        
        # Check that version 2.0 is active
        model_version = model_manager.versions["test-model"]["2.0"]
        assert model_version.is_active is True
    
    @pytest.mark.asyncio
    async def test_switch_nonexistent_version(self, model_manager):
        """Test switching to non-existent version"""
        success = await model_manager.switch_version("nonexistent", "1.0")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_route_request(self, model_manager):
        """Test request routing"""
        # Register a model
        await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        # Route with specific model
        routing_rules = {"model": "test-model", "version": "1.0"}
        model_id = await model_manager.route_request(None, routing_rules)
        assert model_id == "test-model:1.0"
    
    @pytest.mark.asyncio
    async def test_route_request_no_models(self, model_manager):
        """Test routing when no models are available"""
        with pytest.raises(RuntimeError, match="No active models available"):
            await model_manager.route_request(None, {})
    
    @pytest.mark.asyncio
    async def test_get_manager_stats(self, model_manager):
        """Test getting manager statistics"""
        # Register a model
        await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        stats = await model_manager.get_manager_stats()
        
        assert "total_models" in stats
        assert "loaded_models" in stats
        assert "active_models" in stats
        assert stats["total_models"] == 1
        assert stats["active_models"] == 1
    
    @pytest.mark.asyncio
    async def test_get_model_stats(self, model_manager):
        """Test getting model statistics"""
        model_id = await model_manager.register_model(
            name="test-model",
            version="1.0",
            path="test/path",
            format=ModelFormat.HUGGINGFACE
        )
        
        stats = await model_manager.get_model_stats(model_id)
        
        assert "name" in stats
        assert "version" in stats
        assert "state" in stats
        assert stats["name"] == "test-model"
        assert stats["version"] == "1.0"
        assert stats["state"] == ModelState.UNLOADED.value
    
    @pytest.mark.asyncio
    async def test_memory_constraints(self, model_manager):
        """Test memory constraint enforcement"""
        # This is a basic test - in practice, you'd need actual models
        # to test memory constraints properly
        assert model_manager.max_models == 3
        assert model_manager.max_memory_gb == 8
    
    @pytest.mark.asyncio
    async def test_lazy_loading_flag(self, model_manager):
        """Test lazy loading configuration"""
        assert model_manager.enable_lazy_loading is True
        assert model_manager.enable_model_caching is True 