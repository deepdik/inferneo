"""
Performance metrics calculation utilities for benchmarks.
"""

import numpy as np
import statistics
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import time


@dataclass
class LatencyMetrics:
    """Latency performance metrics."""
    p50: float
    p95: float
    p99: float
    p99_9: float
    mean: float
    min: float
    max: float
    std: float
    count: int


@dataclass
class ThroughputMetrics:
    """Throughput performance metrics."""
    requests_per_second: float
    total_requests: int
    success_rate: float
    error_rate: float
    timeout_rate: float
    avg_latency: float


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    peak_memory_mb: float
    average_memory_mb: float
    memory_per_request_mb: float
    memory_efficiency: float


def calculate_latency_metrics(latencies: List[float]) -> LatencyMetrics:
    """Calculate comprehensive latency metrics from a list of latency measurements."""
    if not latencies:
        return LatencyMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    latencies = sorted(latencies)
    count = len(latencies)
    
    return LatencyMetrics(
        p50=np.percentile(latencies, 50),
        p95=np.percentile(latencies, 95),
        p99=np.percentile(latencies, 99),
        p99_9=np.percentile(latencies, 99.9),
        mean=np.mean(latencies),
        min=np.min(latencies),
        max=np.max(latencies),
        std=np.std(latencies),
        count=count
    )


def calculate_throughput_metrics(
    total_requests: int,
    successful_requests: int,
    failed_requests: int,
    timeout_requests: int,
    total_time_seconds: float,
    avg_latency: float
) -> ThroughputMetrics:
    """Calculate throughput metrics."""
    if total_time_seconds <= 0:
        return ThroughputMetrics(0, 0, 0, 0, 0, 0)
    
    rps = total_requests / total_time_seconds
    success_rate = successful_requests / total_requests if total_requests > 0 else 0
    error_rate = failed_requests / total_requests if total_requests > 0 else 0
    timeout_rate = timeout_requests / total_requests if total_requests > 0 else 0
    
    return ThroughputMetrics(
        requests_per_second=rps,
        total_requests=total_requests,
        success_rate=success_rate,
        error_rate=error_rate,
        timeout_rate=timeout_rate,
        avg_latency=avg_latency
    )


def calculate_memory_metrics(
    memory_samples: List[float],
    total_requests: int
) -> MemoryMetrics:
    """Calculate memory usage metrics."""
    if not memory_samples:
        return MemoryMetrics(0, 0, 0, 0)
    
    peak_memory = max(memory_samples)
    average_memory = np.mean(memory_samples)
    memory_per_request = average_memory / total_requests if total_requests > 0 else 0
    memory_efficiency = 1.0 / (1.0 + np.std(memory_samples) / average_memory) if average_memory > 0 else 0
    
    return MemoryMetrics(
        peak_memory_mb=peak_memory,
        average_memory_mb=average_memory,
        memory_per_request_mb=memory_per_request,
        memory_efficiency=memory_efficiency
    )


def calculate_batch_efficiency(
    batch_latencies: List[float],
    batch_sizes: List[int]
) -> Dict[str, float]:
    """Calculate batch processing efficiency metrics."""
    if not batch_latencies or not batch_sizes:
        return {}
    
    # Calculate throughput per batch size
    batch_throughputs = []
    for latency, batch_size in zip(batch_latencies, batch_sizes):
        if latency > 0:
            throughput = batch_size / latency
            batch_throughputs.append(throughput)
    
    if not batch_throughputs:
        return {}
    
    return {
        'avg_batch_throughput': np.mean(batch_throughputs),
        'max_batch_throughput': np.max(batch_throughputs),
        'batch_efficiency': np.mean(batch_throughputs) / np.max(batch_throughputs) if np.max(batch_throughputs) > 0 else 0
    }


def calculate_concurrency_metrics(
    concurrency_levels: List[int],
    latencies: List[List[float]],
    throughputs: List[float]
) -> Dict[str, Any]:
    """Calculate metrics across different concurrency levels."""
    if not concurrency_levels or not latencies:
        return {}
    
    # Calculate optimal concurrency
    max_throughput_idx = np.argmax(throughputs) if throughputs else 0
    optimal_concurrency = concurrency_levels[max_throughput_idx] if max_throughput_idx < len(concurrency_levels) else 0
    
    # Calculate scaling efficiency
    scaling_efficiency = []
    for i, (concurrency, throughput) in enumerate(zip(concurrency_levels, throughputs)):
        if i == 0 or concurrency_levels[i-1] == 0:
            scaling_efficiency.append(1.0)
        else:
            expected_throughput = throughputs[i-1] * (concurrency / concurrency_levels[i-1])
            actual_efficiency = throughput / expected_throughput if expected_throughput > 0 else 0
            scaling_efficiency.append(actual_efficiency)
    
    return {
        'optimal_concurrency': optimal_concurrency,
        'max_throughput': max(throughputs) if throughputs else 0,
        'scaling_efficiency': scaling_efficiency,
        'avg_scaling_efficiency': np.mean(scaling_efficiency) if scaling_efficiency else 0
    }


def calculate_error_metrics(
    total_requests: int,
    errors: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate error analysis metrics."""
    if not errors:
        return {
            'total_errors': 0,
            'error_rate': 0.0,
            'error_types': {},
            'avg_error_latency': 0.0
        }
    
    error_types = {}
    error_latencies = []
    
    for error in errors:
        error_type = error.get('type', 'unknown')
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if 'latency' in error:
            error_latencies.append(error['latency'])
    
    return {
        'total_errors': len(errors),
        'error_rate': len(errors) / total_requests if total_requests > 0 else 0,
        'error_types': error_types,
        'avg_error_latency': np.mean(error_latencies) if error_latencies else 0.0
    }


def calculate_percentile_latency(latencies: List[float], percentile: float) -> float:
    """Calculate a specific percentile latency."""
    if not latencies:
        return 0.0
    return np.percentile(latencies, percentile)


def calculate_confidence_interval(
    data: List[float],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """Calculate confidence interval for the mean."""
    if not data:
        return (0.0, 0.0)
    
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    
    # Z-score for confidence level (simplified)
    z_score = 1.96 if confidence_level == 0.95 else 2.58 if confidence_level == 0.99 else 1.645
    
    margin_of_error = z_score * std_err
    
    return (mean - margin_of_error, mean + margin_of_error) 