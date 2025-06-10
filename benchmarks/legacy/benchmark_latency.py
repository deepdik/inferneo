"""
Latency benchmarks for Inferneo

Measures and compares inference latency across different models and configurations.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any
import argparse
import json

from inferneo.core.enhanced_engine import EnhancedInferneoEngine
from inferneo.core.enhanced_engine import EnhancedEngineConfig
from inferneo.models.base import ModelFormat


class LatencyBenchmark:
    """Benchmark class for measuring inference latency"""
    
    def __init__(self, config: EnhancedEngineConfig):
        self.config = config
        self.engine = None
        self.results = {}
        
    async def setup(self):
        """Setup the benchmark environment"""
        print("Setting up Inferneo engine for benchmarking...")
        self.engine = EnhancedInferneoEngine(self.config)
        await self.engine.start()
        # Register a dummy model for benchmarking
        await self.engine.register_model(
            name="gpt2",
            version="1.0",
            path="gpt2",
            format=ModelFormat.HUGGINGFACE
        )
        
    async def teardown(self):
        """Cleanup benchmark environment"""
        if self.engine:
            await self.engine.stop()
            
    async def benchmark_single_request(self, prompt: str, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark single request latency"""
        print(f"Benchmarking single request latency with {num_runs} runs...")
        
        latencies = []
        for i in range(num_runs):
            start_time = time.time()
            result = await self.engine.generate(prompt)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")
                
        # Calculate statistics
        stats = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95": sorted(latencies)[int(0.95 * len(latencies))],
            "p99": sorted(latencies)[int(0.99 * len(latencies))],
            "num_runs": num_runs
        }
        
        return stats
        
    async def benchmark_batch_requests(self, prompts: List[str], num_runs: int = 50) -> Dict[str, Any]:
        """Benchmark batch request latency"""
        print(f"Benchmarking batch request latency with {num_runs} runs...")
        
        latencies = []
        for i in range(num_runs):
            start_time = time.time()
            results = await self.engine.generate_batch(prompts)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_runs} runs")
                
        # Calculate statistics
        stats = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95": sorted(latencies)[int(0.95 * len(latencies))],
            "p99": sorted(latencies)[int(0.99 * len(latencies))],
            "num_runs": num_runs,
            "batch_size": len(prompts)
        }
        
        return stats
        
    async def benchmark_concurrent_requests(self, prompt: str, num_concurrent: int = 10, num_runs: int = 20) -> Dict[str, Any]:
        """Benchmark concurrent request latency"""
        print(f"Benchmarking concurrent request latency with {num_concurrent} concurrent requests, {num_runs} runs...")
        
        latencies = []
        for run in range(num_runs):
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for _ in range(num_concurrent):
                task = asyncio.create_task(self.engine.generate(prompt))
                tasks.append(task)
                
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            
            if (run + 1) % 5 == 0:
                print(f"Completed {run + 1}/{num_runs} runs")
                
        # Calculate statistics
        stats = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95": sorted(latencies)[int(0.95 * len(latencies))],
            "p99": sorted(latencies)[int(0.99 * len(latencies))],
            "num_runs": num_runs,
            "num_concurrent": num_concurrent
        }
        
        return stats
        
    def print_results(self, results: Dict[str, Any], title: str):
        """Print benchmark results in a formatted way"""
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key:15}: {value:.2f} ms")
            else:
                print(f"{key:15}: {value}")
                
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


async def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Inferneo Latency Benchmark")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf", help="Model to benchmark")
    parser.add_argument("--prompt", default="Hello, how are you?", help="Test prompt")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for batch benchmarks")
    parser.add_argument("--num-concurrent", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--output", default="latency_benchmark_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EnhancedEngineConfig(
        model=None,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        max_waiting_tokens=4096,
        block_size=128
    )
    
    # Create benchmark instance
    benchmark = LatencyBenchmark(config)
    
    try:
        # Setup
        await benchmark.setup()
        
        # Run benchmarks
        results = {}
        
        # Single request benchmark
        single_results = await benchmark.benchmark_single_request(args.prompt, args.num_runs)
        benchmark.print_results(single_results, "Single Request Latency")
        results["single_request"] = single_results
        
        # Batch request benchmark
        batch_prompts = [args.prompt] * args.batch_size
        batch_results = await benchmark.benchmark_batch_requests(batch_prompts, args.num_runs // 2)
        benchmark.print_results(batch_results, "Batch Request Latency")
        results["batch_request"] = batch_results
        
        # Concurrent request benchmark
        concurrent_results = await benchmark.benchmark_concurrent_requests(args.prompt, args.num_concurrent, args.num_runs // 5)
        benchmark.print_results(concurrent_results, "Concurrent Request Latency")
        results["concurrent_request"] = concurrent_results
        
        # Save results
        benchmark.save_results(results, args.output)
        
    finally:
        # Teardown
        await benchmark.teardown()


if __name__ == "__main__":
    asyncio.run(main()) 