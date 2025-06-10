# Inferneo

A high-performance LLM inference engine with advanced features like smart batching, CUDA graph optimization, PagedAttention memory management, speculative decoding, multi-level caching, async-first architecture, plugin system, hot reloading, distributed readiness, and production features such as health monitoring and rate limiting.

## Features

- **Smart Batching**: Dynamic batch size optimization based on request patterns
- **CUDA Graph Optimization**: Pre-compiled CUDA graphs for maximum GPU utilization
- **PagedAttention**: Efficient memory management for long sequences
- **Speculative Decoding**: Parallel token generation for faster inference
- **Multi-Level Caching**: Model, KV, and result caching for optimal performance
- **Async-First Architecture**: Non-blocking I/O and concurrent request processing
- **Plugin System**: Extensible architecture for custom optimizations
- **Hot Reloading**: Model updates without service interruption
- **Distributed Readiness**: Multi-GPU and multi-node support
- **Production Features**: Health monitoring, rate limiting, and metrics

## Installation

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core and development dependencies
pip install -r requirements.txt
```
> **Note:** Some dependencies (e.g., `auto-gptq`, `tensorrt`) are optional and may require CUDA or special hardware. If you encounter issues, comment them out in `requirements.txt` and install only what you need.

## Running Tests

You can run all tests and checks using the test runner:

```bash
python tests/run_tests.py
```

Or run specific test types:

```bash
python tests/run_tests.py --unit         # Unit tests
python tests/run_tests.py --integration  # Integration tests
python tests/run_tests.py --performance  # Performance tests
python tests/run_tests.py --coverage     # Coverage report
python tests/run_tests.py --lint         # Linting
python tests/run_tests.py --type-check   # Type checking
python tests/run_tests.py --benchmarks   # Run benchmarks
python tests/run_tests.py --all          # Run all tests and checks
```

You can also use `pytest` directly:

```bash
pytest tests/ -v
```

## Running Benchmarks

Benchmarks are managed via scripts in `benchmarks/runners/`. Example commands:

```bash
# Run Inferneo latency benchmark
python -m benchmarks.runners.inferneo_runner --scenario latency --model gpt2

# Run Triton throughput benchmark
python -m benchmarks.runners.triton_runner --scenario throughput --model distilgpt2

# Run head-to-head comparison
python -m benchmarks.runners.comparison_runner --models gpt2,distilgpt2 --scenarios latency,throughput

# Run comprehensive benchmark suite
python -m benchmarks.runners.comparison_runner --comprehensive
```

See [`benchmarks/README.md`](benchmarks/README.md) for more details and advanced options.

## Examples

See the `examples/` directory for:

- **Basic usage:** `examples/basic/basic_usage.py`
- **Advanced features:** `examples/advanced/`
- **Multi-model serving:** `examples/multi_model/multi_model_serving.py`

## Project Structure

```
inferneo/
├── inferneo/
│   ├── core/
│   │   ├── engine.py         # Main InferneoEngine
│   │   ├── enhanced_engine.py # Enhanced multi-model engine
│   │   ├── config.py         # Configuration management
│   │   ├── scheduler.py      # Request scheduling
│   │   ├── memory_manager.py # Memory management
│   │   └── cache_manager.py  # Caching system
│   ├── models/
│   │   ├── base.py          # Base model interface
│   │   ├── manager.py       # Model manager with lazy loading
│   │   ├── registry.py      # Model registry
│   │   ├── transformers.py  # HuggingFace models
│   │   ├── onnx/           # ONNX model support
│   │   └── tensorrt/       # TensorRT model support
│   └── __init__.py
├── benchmarks/
│   ├── runners/
│   ├── scenarios/
│   ├── utils/
│   └── ...
├── examples/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
└── docs/                 # Documentation
```

## Basic Usage

```python
from inferneo.core.engine import InferneoEngine
from inferneo.core.config import EngineConfig

# Configure the engine
config = EngineConfig(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_model_len=4096,
    gpu_memory_utilization=0.9
)

# Initialize and start the engine
engine = InferneoEngine(config)
engine.start()

# Generate text
result = engine.generate(
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7
)

print(result.text)
engine.stop()
```

## Performance Comparison

| Feature | Inferneo | vLLM | NVIDIA Triton |
|---------|----------|------|---------------|
| Smart Batching | ✅ | ✅ | ✅ |
| CUDA Graphs | ✅ | ✅ | ✅ |
| PagedAttention | ✅ | ✅ | ❌ |
| Speculative Decoding | ✅ | ✅ | ❌ |
| Multi-Model Serving | ✅ | ❌ | ✅ |
| Hot Reloading | ✅ | ❌ | ✅ |
| Plugin System | ✅ | ❌ | ✅ |
| Async Architecture | ✅ | ✅ | ✅ |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

---

**Inferneo** - Accelerating AI inference to new heights! 🚀
