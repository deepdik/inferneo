
 # Dynamic Batching for Inferneo

## 🚀 Overview

The Enhanced Dynamic Batching Scheduler is a high-performance request scheduling system designed to optimize LLM inference throughput and latency. It provides intelligent batching strategies, adaptive sizing, and real-time performance monitoring.

## ✨ Key Features

### 🎯 Multiple Batching Strategies
- **Adaptive Batching**: Automatically adjusts batch size based on GPU utilization and performance metrics
- **Coalescing Batching**: Groups similar-length requests for optimal memory alignment
- **Latency-Optimized**: Prioritizes low latency with smaller batches
- **Throughput-Optimized**: Maximizes throughput with larger batches
- **Fixed-Size**: Traditional fixed-size batching for comparison

### 📊 Real-Time Performance Monitoring
- GPU utilization tracking
- Batch performance metrics
- Adaptive batch size adjustment
- Queue depth monitoring
- Latency and throughput statistics

### 🔧 Intelligent Request Management
- Priority-based queuing
- Request coalescing by length
- Dynamic timeout adjustment
- Background metrics collection
- Performance-based optimization

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Request       │    │   Enhanced       │    │   Batch         │
│   Queue         │───▶│   Scheduler      │───▶│   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Performance    │
                       │   Monitor        │
                       └──────────────────┘
```

## 📦 Installation

The enhanced scheduler is included in the Inferneo package. No additional installation is required.

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from inferneo.core.enhanced_scheduler import EnhancedScheduler, BatchingStrategy
from inferneo.core.config import SchedulerConfig

# Configure scheduler
config = SchedulerConfig(
    max_num_partial_prefills=4,
    max_waiting_tokens=2048,
    enable_priority_queue=True
)

# Create scheduler
scheduler = EnhancedScheduler(config)

# Initialize
await scheduler.initialize()

# Set batching strategy
scheduler.set_batching_strategy(BatchingStrategy.ADAPTIVE)

# Add requests
request = DemoRequest(
    request_id="req_001",
    prompt="Hello, world!",
    max_tokens=50,
    priority=5
)
await scheduler.add_request(request)

# Get next batch
batch = await scheduler.get_next_batch()

# Process batch and complete
processing_time = 0.1
gpu_utilization = 0.8
memory_usage = 0.6

await scheduler.complete_batch(batch[0].batch_id, processing_time, gpu_utilization, memory_usage)

# Cleanup
await scheduler.cleanup()
```

### Running the Demo

```bash
cd inferneo/examples
python dynamic_batching_demo.py
```

## 🎛️ Configuration

### SchedulerConfig Options

```python
SchedulerConfig(
    # Maximum number of requests in a batch
    max_num_partial_prefills=4,
    
    # Maximum tokens per batch
    max_waiting_tokens=2048,
    
    # Enable priority-based queuing
    enable_priority_queue=True,
    
    # Preemption mode for high-priority requests
    preemption_mode="recompute",  # or "swap"
    
    # Enable chunked prefill
    enable_chunked_prefill=True,
    
    # Long prefill token threshold
    long_prefill_token_threshold=8192
)
```

## 📈 Batching Strategies

### 1. Adaptive Batching (Recommended)

Automatically adjusts batch size based on:
- Current GPU utilization
- Recent performance metrics
- Queue depth
- Request characteristics

```python
scheduler.set_batching_strategy(BatchingStrategy.ADAPTIVE)
```

**Benefits:**
- Optimal performance across different workloads
- Automatic adaptation to system conditions
- Balanced latency and throughput

### 2. Coalescing Batching

Groups requests with similar prompt lengths for optimal memory alignment:

```python
scheduler.set_batching_strategy(BatchingStrategy.COALESCING)
```

**Benefits:**
- Reduced padding overhead
- Better memory utilization
- Improved GPU efficiency

### 3. Latency-Optimized Batching

Prioritizes low latency with smaller batches:

```python
scheduler.set_batching_strategy(BatchingStrategy.LATENCY_OPTIMIZED)
```

**Benefits:**
- Lower response times
- Better for interactive applications
- Reduced queue wait times

### 4. Throughput-Optimized Batching

Maximizes throughput with larger batches:

```python
scheduler.set_batching_strategy(BatchingStrategy.THROUGHPUT_OPTIMIZED)
```

**Benefits:**
- Higher requests per second
- Better GPU utilization
- Optimal for batch processing

## 📊 Performance Monitoring

### Real-Time Metrics

```python
# Get comprehensive statistics
stats = scheduler.get_stats()

print(f"Total Requests: {stats['total_requests']}")
print(f"Queue Depth: {stats['queue_depth']}")
print(f"Average Wait Time: {stats['avg_wait_time']:.3f}s")
print(f"Average Processing Time: {stats['avg_processing_time']:.3f}s")

# Get batching-specific statistics
batching_stats = scheduler.get_batching_stats()

print(f"Current Batch Size: {batching_stats['current_batch_size']}")
print(f"Average Throughput: {batching_stats['avg_throughput']:.2f} req/s")
print(f"Average Latency: {batching_stats['avg_latency']:.3f}s")
print(f"GPU Utilization: {batching_stats['avg_gpu_utilization']:.2f}")
```

### Performance Metrics

The scheduler tracks:
- **Batch Size**: Number of requests per batch
- **Throughput**: Requests processed per second
- **Latency**: Average processing time per request
- **GPU Utilization**: Current GPU usage percentage
- **Memory Usage**: Memory consumption during processing
- **Queue Depth**: Number of waiting requests

## 🔧 Advanced Features

### Custom Request Priorities

```python
# High priority request
high_priority_request = DemoRequest(
    request_id="urgent_001",
    prompt="Emergency response needed",
    max_tokens=20,
    priority=10  # Highest priority
)

# Low priority request
low_priority_request = DemoRequest(
    request_id="background_001",
    prompt="Background processing",
    max_tokens=100,
    priority=1   # Lowest priority
)
```

### Dynamic Strategy Switching

```python
# Switch strategies based on workload
if high_latency_workload:
    scheduler.set_batching_strategy(BatchingStrategy.LATENCY_OPTIMIZED)
elif high_throughput_workload:
    scheduler.set_batching_strategy(BatchingStrategy.THROUGHPUT_OPTIMIZED)
else:
    scheduler.set_batching_strategy(BatchingStrategy.ADAPTIVE)
```

### Request Cancellation

```python
# Cancel a specific request
success = await scheduler.cancel_request("req_001")
if success:
    print("Request cancelled successfully")
```

## 📈 Performance Comparison

### Expected Improvements

| Strategy | Throughput | Latency | GPU Utilization | Use Case |
|----------|------------|---------|-----------------|----------|
| Fixed-Size | Baseline | Baseline | Baseline | Reference |
| Adaptive | +40-60% | -20-30% | +15-25% | General purpose |
| Coalescing | +30-50% | -10-20% | +20-30% | Similar-length requests |
| Latency-Optimized | -20-30% | -40-60% | -10-20% | Interactive applications |
| Throughput-Optimized | +60-80% | +20-40% | +30-40% | Batch processing |

### Benchmark Results

Run the demo to see real performance metrics:

```bash
python dynamic_batching_demo.py
```

Example output:
```
📊 Demo 1: Adaptive Batching Strategy
==================================================
Adaptive Batching Results:
  - Average Batch Size: 3.45
  - Average Throughput: 12.34 req/s
  - Average Latency: 0.234s

🔗 Demo 2: Coalescing Batching Strategy
==================================================
Coalescing Batching Results:
  - Average Batch Size: 4.12
  - Coalescing Buckets: 3
  - Average Throughput: 15.67 req/s
```

## 🛠️ Integration with Enhanced Engine

The enhanced scheduler can be integrated with the EnhancedInferneoEngine:

```python
from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig

# Configure engine with enhanced scheduler
config = EnhancedEngineConfig(
    model="gpt2",
    max_workers=4,
    enable_dynamic_batching=True,
    enable_speculative_decoding=True,
    enable_kv_cache_optimization=True
)

# Create engine
engine = EnhancedInferneoEngine(config)

# The engine will automatically use the enhanced scheduler
await engine.initialize()
```

## 🔍 Troubleshooting

### Common Issues

1. **High Latency**
   - Switch to Latency-Optimized strategy
   - Reduce batch size
   - Check GPU utilization

2. **Low Throughput**
   - Switch to Throughput-Optimized strategy
   - Increase batch size
   - Monitor queue depth

3. **Memory Issues**
   - Reduce max_waiting_tokens
   - Use Coalescing strategy
   - Monitor memory usage

### Debug Information

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

To contribute to the dynamic batching system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by modern inference engines like vLLM and TensorRT-LLM
- GPU monitoring based on PyNVML
- Performance metrics inspired by industry best practices