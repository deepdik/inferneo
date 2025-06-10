# Inferneo - Phase 1 Implementation

Phase 1 of the Inferneo project implements the core multi-model serving infrastructure with lazy loading, multi-format support, and an enhanced engine for production-ready inference.

## Overview

Phase 1 focuses on building a solid foundation for high-performance LLM inference with the following key components:

1. **ModelManager with Lazy Loading**: Efficient model loading and memory management
2. **Multi-Format Support**: HuggingFace, ONNX, and TensorRT models
3. **Enhanced Engine**: Multi-model serving with A/B testing and dynamic routing
4. **Comprehensive Testing**: Unit, integration, and performance tests
5. **Benchmarking Tools**: Latency and performance measurement

## Architecture

### Core Components

#### 1. ModelManager (`inferneo/models/manager.py`)

The ModelManager provides advanced model management capabilities:

- **Lazy Loading**: Models are loaded only when needed
- **Versioning**: Support for multiple model versions
- **Multi-Format Support**: HuggingFace, ONNX, TensorRT
- **Memory Management**: Automatic cleanup and memory optimization
- **Registry Persistence**: Model registry saved to disk

```python
from inferneo.models.manager import ModelManager

manager = ModelManager(
    max_models=5,
    max_memory_gb=16.0,
    enable_lazy_loading=True
)

# Register a model
model_id = await manager.register_model(
    name="gpt-3.5-turbo",
    version="1.0",
    path="models/gpt-3.5-turbo",
    format=ModelFormat.HUGGINGFACE
)

# Load model (lazy loading)
model = await manager.load_model(model_id)
```

#### 2. Enhanced Engine (`inferneo/core/enhanced_engine.py`)

The EnhancedInferneoEngine extends the base engine with multi-model capabilities:

- **Multi-Model Serving**: Serve multiple models simultaneously
- **A/B Testing**: Compare different models and configurations
- **Dynamic Routing**: Route requests based on content and rules
- **Performance Monitoring**: Real-time metrics and health checks

```python
from inferneo.core.enhanced_engine import EnhancedInferneoEngine, RoutingRule, ABTestConfig

engine = EnhancedInferneoEngine(config)

# Add routing rules
rule = RoutingRule(
    name="code_routing",
    condition="prompt contains 'code'",
    model_name="code-llama",
    priority=10
)
engine.add_routing_rule(rule)

# Add A/B test
ab_test = ABTestConfig(
    name="model_comparison",
    model_a="gpt-3.5-turbo",
    model_b="code-llama",
    traffic_split=0.3
)
engine.add_ab_test(ab_test)
```

#### 3. Multi-Format Model Support

##### HuggingFace Models (`inferneo/models/transformers.py`)

Full support for HuggingFace Transformers models with optimizations:

```python
from inferneo.models.transformers import TransformersModel

model = TransformersModel("meta-llama/Llama-2-7b-chat-hf", config)
await model.initialize(config)
result = await model.generate("Hello, world!")
```

##### ONNX Models (`inferneo/models/onnx/`)

ONNX model support with conversion utilities:

```python
from inferneo.models.onnx import ONNXModel, ONNXConverter

# Convert HuggingFace model to ONNX
converter = ONNXConverter()
converter.convert_model("meta-llama/Llama-2-7b-chat-hf", "model.onnx")

# Use ONNX model
model = ONNXModel("model.onnx", config)
await model.initialize(config)
```

##### TensorRT Models (`inferneo/models/tensorrt/`)

TensorRT optimized models for maximum performance:

```python
from inferneo.models.tensorrt import TensorRTModel

model = TensorRTModel("model.trt", config)
await model.initialize(config)
```

## Testing Infrastructure

### Unit Tests (`tests/unit/`)

Comprehensive unit tests for all components:

- **`test_model_manager.py`**: ModelManager functionality
- **`test_enhanced_engine.py`**: Enhanced engine features
- **`test_engine.py`**: Base engine functionality

### Integration Tests (`tests/integration/`)

End-to-end tests for complete workflows:

- **`test_phase1_integration.py`**: Phase 1 integration tests

### Test Runner (`tests/run_tests.py`)

Comprehensive test runner with multiple options:

```bash
# Run all tests
python tests/run_tests.py --all

# Run specific test types
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --coverage
```

## Benchmarking

### Latency Benchmarks (`benchmarks/latency/`)

Performance measurement tools:

```bash
# Run latency benchmarks
python benchmarks/latency/benchmark_latency.py --num-runs=100
```

The benchmark suite includes:
- Single request latency
- Batch request latency
- Concurrent request latency
- Statistical analysis (mean, median, p95, p99)

## Examples

### Basic Usage (`examples/basic/basic_usage.py`)

Simple example demonstrating core functionality:

```python
from inferneo.core.engine import InferneoEngine
from inferneo.core.config import EngineConfig

config = EngineConfig(model="meta-llama/Llama-2-7b-chat-hf")
engine = InferneoEngine(config)

await engine.start()
result = await engine.generate("Hello, world!")
await engine.stop()
```

### Multi-Model Serving (`examples/multi_model/multi_model_serving.py`)

Advanced example showing multi-model capabilities:

```python
from inferneo.core.enhanced_engine import EnhancedInferneoEngine

engine = EnhancedInferneoEngine(config)

# Register multiple models
await engine.register_model("gpt-3.5-turbo", "1.0", "models/gpt-3.5-turbo")
await engine.register_model("code-llama", "1.0", "models/code-llama")

# Add routing and A/B testing
engine.add_routing_rule(code_rule)
engine.add_ab_test(ab_test)

# Generate with automatic routing
result = await engine.generate("Write a Python function")
```

## Configuration

The project uses a comprehensive configuration system:

```python
from inferneo.core.config import EngineConfig

config = EngineConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    enable_cuda_graph=True,
    enable_paged_attention=True,
    enable_speculative_decoding=True
)
```

Environment variables are also supported:

```bash
export INFERNEO_MODEL="meta-llama/Llama-2-7b-chat-hf"
export INFERNEO_MAX_MODEL_LEN=4096
export INFERNEO_GPU_MEMORY_UTIL=0.9
```

## Key Features Implemented

### 1. Lazy Loading

Models are loaded only when needed, reducing memory usage and startup time:

```python
# Model is not loaded until first use
model = await manager.get_model("gpt-3.5-turbo")
```

### 2. Multi-Format Support

Support for multiple model formats with unified interface:

- **HuggingFace**: Full Transformers support
- **ONNX**: Optimized inference with conversion tools
- **TensorRT**: Maximum performance optimization

### 3. A/B Testing

Compare different models and configurations:

```python
ab_test = ABTestConfig(
    name="model_comparison",
    model_a="gpt-3.5-turbo",
    model_b="code-llama",
    traffic_split=0.3
)
```

### 4. Dynamic Routing

Route requests based on content and rules:

```python
rule = RoutingRule(
    name="code_routing",
    condition="prompt contains 'code'",
    model_name="code-llama",
    priority=10
)
```

### 5. Performance Monitoring

Real-time metrics and health checks:

```python
stats = engine.get_stats()
print(f"Requests per second: {stats['requests_per_second']}")
print(f"Model metrics: {stats['model_metrics']}")
```

## Performance Characteristics

### Memory Management

- **Lazy Loading**: Models loaded on-demand
- **Memory Pooling**: Efficient GPU memory allocation
- **Automatic Cleanup**: Unused models automatically unloaded

### Throughput Optimization

- **Smart Batching**: Dynamic batch size optimization
- **Concurrent Processing**: Async-first architecture
- **Caching**: Multi-level caching for improved performance

### Latency Optimization

- **CUDA Graphs**: Pre-compiled CUDA graphs for repeated operations
- **PagedAttention**: Efficient memory management for long sequences
- **Speculative Decoding**: Parallel token generation

## Development Workflow

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run with coverage
python tests/run_tests.py --coverage

# Run specific test types
python tests/run_tests.py --unit --integration
```

### Running Examples

```bash
# Basic usage
python examples/basic/basic_usage.py

# Multi-model serving
python examples/multi_model/multi_model_serving.py
```

### Running Benchmarks

```bash
# Latency benchmarks
python benchmarks/latency/benchmark_latency.py --num-runs=100
```

## Next Steps (Phase 2)

Phase 2 will build upon Phase 1 with:

1. **C/C++ Extensions**: Native performance optimizations
2. **Advanced Optimizations**: CUDA kernels, custom attention mechanisms
3. **Production Features**: Load balancing, distributed serving
4. **Monitoring**: Advanced metrics, alerting, observability
5. **Deployment**: Docker, Kubernetes, cloud deployment

## Contributing

1. Follow the established architecture patterns
2. Add comprehensive tests for new features
3. Update documentation
4. Run the full test suite before submitting

## Conclusion

Phase 1 provides a solid foundation for high-performance LLM inference with advanced features like multi-model serving, A/B testing, and dynamic routing. The modular architecture makes it easy to extend and customize for specific use cases. 