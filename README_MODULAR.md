# Inferneo - Modular Structure

This document describes the modular structure of the Inferneo project, which is designed for high-performance LLM inference with advanced features like multi-model serving, A/B testing, and dynamic routing.

## 📁 Project Structure

```
inferneo/
├── inferneo/                    # Main package
│   ├── __init__.py             # Package initialization
│   ├── core/                   # Core engine components
│   │   ├── __init__.py
│   │   ├── engine.py           # Main InferneoEngine
│   │   ├── enhanced_engine.py  # Enhanced multi-model engine
│   │   ├── config.py           # Configuration management
│   │   ├── scheduler.py        # Request scheduling
│   │   ├── memory_manager.py   # Memory management
│   │   └── cache_manager.py    # Caching system
│   ├── models/                 # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py            # Base model interface
│   │   ├── manager.py         # Model manager with lazy loading
│   │   ├── registry.py        # Model registry
│   │   ├── transformers.py    # HuggingFace models
│   │   ├── onnx/             # ONNX model support
│   │   │   ├── __init__.py
│   │   │   ├── onnx_model.py
│   │   │   └── converter.py
│   │   └── tensorrt/         # TensorRT model support
│   │       ├── __init__.py
│   │       └── tensorrt_model.py
│   ├── utils/                   # Utility modules
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── tests/                    # Test suite
│   │   ├── __init__.py
│   │   ├── run_tests.py         # Test runner
│   │   ├── unit/                # Unit tests
│   │   │   ├── __init__.py
│   │   │   ├── test_engine.py
│   │   │   ├── test_model_manager.py
│   │   │   └── test_enhanced_engine.py
│   │   ├── integration/         # Integration tests
│   │   │   ├── __init__.py
│   │   │   └── test_phase1_integration.py
│   │   └── performance/         # Performance tests
│   │       └── test_performance.py
│   ├── benchmarks/              # Performance benchmarks
│   │   ├── __init__.py
│   │   ├── latency/            # Latency benchmarks
│   │   │   ├── __init__.py
│   │   │   └── benchmark_latency.py
│   │   ├── throughput/         # Throughput benchmarks
│   │   └── memory/            # Memory usage benchmarks
│   ├── examples/                # Usage examples
│   │   ├── __init__.py
│   │   ├── basic/              # Basic usage examples
│   │   │   ├── __init__.py
│   │   │   └── basic_usage.py
│   │   ├── advanced/           # Advanced features
│   │   └── multi_model/        # Multi-model serving
│   │       ├── __init__.py
│   │       └── multi_model_serving.py
│   ├── docs/                   # Documentation
│   │   ├── api/                # API documentation
│   │   ├── guides/             # User guides
│   │   └── architecture/       # Architecture docs
│   ├── scripts/                # Utility scripts
│   │   ├── setup.py            # Setup script
│   │   └── deploy.py           # Deployment script
│   ├── requirements.txt        # Dependencies
│   ├── pytest.ini            # Pytest configuration
│   ├── README.md             # Main README
│   ├── README_MODULAR.md     # This file
│   └── PHASE1_README.md      # Phase 1 implementation details
└── PHASE1_SUMMARY.md         # Phase 1 summary
```

## 🧪 Testing Structure

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Coverage**: ModelManager, EnhancedEngine, individual model implementations
- **Dependencies**: Minimal, use mocks for external dependencies
- **Speed**: Fast execution

```bash
# Run unit tests only
python tests/run_tests.py --unit

# Run with pytest directly
pytest tests/unit/ -v
```

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions and workflows
- **Coverage**: End-to-end workflows, multi-model scenarios
- **Dependencies**: May require actual models (with fallbacks)
- **Speed**: Medium execution time

```bash
# Run integration tests only
python tests/run_tests.py --integration

# Run with pytest directly
pytest tests/integration/ -v -m integration
```

### Performance Tests (`tests/performance/`)
- **Purpose**: Test performance characteristics
- **Coverage**: Latency, throughput, memory usage
- **Dependencies**: May require GPU and actual models
- **Speed**: Slow execution

```bash
# Run performance tests
pytest tests/performance/ -v -m performance
```

## 📊 Benchmarking Structure

### Latency Benchmarks (`benchmarks/latency/`)
- **Purpose**: Measure response times
- **Metrics**: Average, P50, P95, P99 latencies
- **Output**: JSON reports, charts

```bash
# Run latency benchmarks
python benchmarks/latency/benchmark_latency.py
```

### Throughput Benchmarks (`benchmarks/throughput/`)
- **Purpose**: Measure requests per second
- **Metrics**: RPS, concurrent requests, batch processing
- **Output**: Performance reports

### Memory Benchmarks (`benchmarks/memory/`)
- **Purpose**: Measure memory usage
- **Metrics**: GPU memory, CPU memory, memory efficiency
- **Output**: Memory analysis reports

## 📚 Examples Structure

### Basic Examples (`examples/basic/`)
- **Purpose**: Simple usage examples
- **Audience**: Beginners, quick start
- **Content**: Single model, basic generation

```bash
# Run basic example
python examples/basic/basic_usage.py
```

### Advanced Examples (`examples/advanced/`)
- **Purpose**: Advanced feature demonstrations
- **Audience**: Experienced users
- **Content**: Custom configurations, optimizations

### Multi-Model Examples (`examples/multi_model/`)
- **Purpose**: Multi-model serving demonstrations
- **Audience**: Production users
- **Content**: A/B testing, routing, version switching

```bash
# Run multi-model example
python examples/multi_model/multi_model_serving.py
```

## 🛠️ Development Workflow

### 1. Setup Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy
```

### 2. Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --benchmarks
python tests/run_tests.py --examples

# Run with pytest directly
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/integration/ -v -m integration
```

### 3. Code Quality

```bash
# Format code
black inferneo/ tests/ benchmarks/ examples/

# Lint code
flake8 inferneo/ tests/ benchmarks/ examples/

# Type checking
mypy inferneo/

# Run all quality checks
python scripts/quality_check.py
```

### 4. Running Examples

```bash
# Run basic example
python examples/basic/basic_usage.py

# Run multi-model example
python examples/multi_model/multi_model_serving.py

# Run benchmarks
python benchmarks/latency/benchmark_latency.py
```

## 📋 Test Categories

### Unit Tests
- **ModelManager**: Registration, loading, routing, versioning
- **EnhancedEngine**: Configuration, routing strategies, A/B testing
- **Model Implementations**: HuggingFace, ONNX, TensorRT
- **Core Components**: Scheduler, memory manager, cache manager

### Integration Tests
- **Full Workflow**: Model registration → generation → statistics
- **Multi-Model**: Concurrent model serving, routing
- **A/B Testing**: Traffic splitting, metrics collection
- **Version Switching**: Hot-swapping model versions
- **Performance Monitoring**: Metrics collection, health checks

### Performance Tests
- **Latency**: Response time under various loads
- **Throughput**: Requests per second capacity
- **Memory**: Memory usage patterns and efficiency
- **Concurrency**: Multi-request handling

## 🎯 Testing Best Practices

### 1. Test Organization
- **Unit tests**: Test one thing at a time
- **Integration tests**: Test component interactions
- **Performance tests**: Test under realistic conditions

### 2. Test Data
- **Mock data**: Use for unit tests
- **Real models**: Use for integration tests (with fallbacks)
- **Benchmark data**: Use for performance tests

### 3. Test Configuration
- **Fast tests**: Unit tests should run quickly
- **Isolated tests**: Tests should not depend on each other
- **Repeatable tests**: Tests should produce consistent results

### 4. Error Handling
- **Graceful failures**: Tests should handle missing dependencies
- **Clear error messages**: Tests should provide helpful feedback
- **Fallback mechanisms**: Tests should work without external resources

## 📈 Benchmarking Best Practices

### 1. Benchmark Configuration
- **Warm-up runs**: Include warm-up before measurement
- **Multiple iterations**: Run multiple times for statistical significance
- **Consistent environment**: Use same hardware/software configuration

### 2. Benchmark Metrics
- **Latency**: Response time percentiles
- **Throughput**: Requests per second
- **Memory**: Peak and average memory usage
- **CPU/GPU**: Utilization metrics

### 3. Benchmark Reporting
- **JSON output**: Machine-readable results
- **Charts**: Visual representation of results
- **Comparisons**: Compare different configurations

## 🔧 Configuration Files

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --asyncio-mode=auto
    --cov=inferneo
    --cov-report=term-missing
    --cov-report=html:htmlcov
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    gpu: Tests requiring GPU
    cpu: CPU-only tests
asyncio_mode = auto
```

### Test Runner
The `tests/run_tests.py` script provides a unified interface for running all types of tests:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test types
python tests/run_tests.py --unit
python tests/run_tests.py --integration
python tests/run_tests.py --benchmarks
python tests/run_tests.py --examples
```

## 🚀 Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run tests
        run: python tests/run_tests.py
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

## 📊 Coverage and Quality

### Code Coverage
- **Target**: >90% code coverage
- **Tools**: pytest-cov, coverage.py
- **Reports**: HTML and terminal output

### Code Quality
- **Formatting**: Black for consistent code style
- **Linting**: Flake8 for code quality checks
- **Type Checking**: MyPy for type safety

## 🎯 Next Steps

1. **Add more unit tests** for edge cases
2. **Expand integration tests** for complex scenarios
3. **Add performance regression tests**
4. **Create automated benchmark comparisons**
5. **Add more examples** for different use cases
6. **Implement continuous benchmarking**

---

This modular structure provides:
- ✅ **Clear separation** of concerns
- ✅ **Comprehensive testing** at multiple levels
- ✅ **Easy benchmarking** and performance analysis
- ✅ **Practical examples** for different use cases
- ✅ **Maintainable codebase** with proper organization
- ✅ **Scalable architecture** for future development 