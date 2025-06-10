# Inferneo Benchmarks

This directory contains comprehensive benchmarks for the Inferneo inference engine, comparing performance with NVIDIA Triton Inference Server and other inference solutions.

## Folder Structure

```
benchmarks/
├── README.md                           # This file
├── config/                             # Benchmark configurations
│   ├── models.json                     # Model configurations
│   ├── scenarios.json                  # Benchmark scenarios
│   └── environments.json               # Environment configurations
├── runners/                            # Benchmark execution scripts
│   ├── __init__.py
│   ├── inferneo_runner.py              # Inferneo benchmark runner
│   ├── triton_runner.py                # Triton benchmark runner
│   └── comparison_runner.py            # Head-to-head comparison runner
├── scenarios/                          # Different benchmark scenarios
│   ├── __init__.py
│   ├── latency/                        # Latency-focused benchmarks
│   │   ├── __init__.py
│   │   ├── single_request.py
│   │   ├── concurrent_requests.py
│   │   └── batch_processing.py
│   ├── throughput/                     # Throughput-focused benchmarks
│   │   ├── __init__.py
│   │   ├── max_throughput.py
│   │   ├── sustained_load.py
│   │   └── burst_handling.py
│   ├── memory/                         # Memory usage benchmarks
│   │   ├── __init__.py
│   │   ├── memory_profiling.py
│   │   ├── memory_efficiency.py
│   │   └── memory_scaling.py
│   └── production/                     # Production-like scenarios
│       ├── __init__.py
│       ├── mixed_workload.py
│       ├── stress_test.py
│       └── real_world_simulation.py
├── utils/                              # Utility functions
│   ├── __init__.py
│   ├── metrics.py                      # Performance metrics calculation
│   ├── reporting.py                    # Report generation utilities
│   ├── visualization.py                # Charts and graphs
│   └── data_processing.py              # Data processing utilities
├── results/                            # Benchmark results storage
│   ├── inferneo/                       # Inferneo-specific results
│   ├── triton/                         # Triton-specific results
│   ├── comparisons/                    # Head-to-head comparison results
│   └── historical/                     # Historical benchmark data
├── reports/                            # Generated reports
│   ├── latest/                         # Latest benchmark reports
│   ├── summaries/                      # Executive summaries
│   └── detailed/                       # Detailed technical reports
└── legacy/                             # Legacy benchmark files (to be cleaned up)
```

## Quick Start

### Running a Single Benchmark

```bash
# Run Inferneo latency benchmark
python -m benchmarks.runners.inferneo_runner --scenario latency --model gpt2

# Run Triton throughput benchmark
python -m benchmarks.runners.triton_runner --scenario throughput --model distilgpt2

# Run head-to-head comparison
python -m benchmarks.runners.comparison_runner --models gpt2,distilgpt2 --scenarios latency,throughput
```

### Running All Benchmarks

```bash
# Run comprehensive benchmark suite
python -m benchmarks.runners.comparison_runner --comprehensive

# Generate report
python -m benchmarks.utils.reporting --generate-summary
```

## Benchmark Types

### 1. Latency Benchmarks
- **Single Request Latency**: Measures end-to-end latency for individual requests
- **Concurrent Request Latency**: Measures latency under concurrent load
- **Batch Processing Latency**: Measures latency for batch inference

### 2. Throughput Benchmarks
- **Maximum Throughput**: Finds the maximum requests per second
- **Sustained Load**: Tests performance under sustained load
- **Burst Handling**: Tests ability to handle traffic spikes

### 3. Memory Benchmarks
- **Memory Profiling**: Detailed memory usage analysis
- **Memory Efficiency**: Memory usage per request
- **Memory Scaling**: How memory usage scales with load

### 4. Production Benchmarks
- **Mixed Workload**: Real-world mixed request patterns
- **Stress Testing**: Extreme load conditions
- **Real-world Simulation**: Production-like traffic patterns

## Models Supported

- **GPT-2**: 124M parameters
- **DistilGPT-2**: 82M parameters
- **Custom Models**: ONNX, TensorRT, PyTorch models

## Metrics Collected

### Performance Metrics
- **Latency**: P50, P95, P99, P99.9 percentiles
- **Throughput**: Requests per second (RPS)
- **Concurrency**: Maximum concurrent requests
- **Batch Efficiency**: Requests per batch

### Resource Metrics
- **Memory Usage**: Peak, average, per-request
- **CPU Usage**: Utilization and efficiency
- **GPU Usage**: Utilization and memory
- **Network**: I/O patterns

### Quality Metrics
- **Accuracy**: Model output quality
- **Reliability**: Error rates and stability
- **Scalability**: Performance scaling characteristics

## Report Generation

Benchmarks automatically generate comprehensive reports including:

1. **Executive Summary**: High-level performance comparison
2. **Detailed Analysis**: Technical deep-dive into results
3. **Visualizations**: Charts and graphs for easy interpretation
4. **Recommendations**: Optimization suggestions
5. **Historical Trends**: Performance over time

## Configuration

Benchmark behavior can be customized through configuration files:

- `config/models.json`: Model specifications and parameters
- `config/scenarios.json`: Benchmark scenario definitions
- `config/environments.json`: Environment-specific settings

## Contributing

When adding new benchmarks:

1. Follow the folder structure
2. Use the utility functions for metrics and reporting
3. Include proper documentation
4. Add configuration options
5. Update this README

## Troubleshooting

Common issues and solutions:

- **Model Loading Errors**: Check model paths and formats
- **Triton Connection Issues**: Verify Triton server status
- **Memory Errors**: Adjust batch sizes and concurrency limits
- **Performance Issues**: Check system resources and configurations 