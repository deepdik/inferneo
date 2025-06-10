#!/usr/bin/env python3
"""
Comprehensive Benchmark: Inferneo vs Triton

This script runs identical benchmarks on both Inferneo and Triton
to provide a fair performance comparison.
"""

import asyncio
import time
import statistics
import json
import requests
import numpy as np
from typing import Dict, List, Any
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig
from inferneo.models.base import ModelFormat
from inferneo.models.manager import ModelMetadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveBenchmark:
    """Comprehensive benchmark comparing Inferneo vs Triton"""
    
    def __init__(self):
        # Create config for EnhancedInferneoEngine
        config = EnhancedEngineConfig(
            model="gpt2",
            max_workers=4,
            max_memory_gb=8,
            enable_dynamic_batching=True,
            enable_speculative_decoding=True,
            enable_kv_cache_optimization=True
        )
        self.inferneo_engine = EnhancedInferneoEngine(config)
        self.triton_url = "http://localhost:8003"
        self.results = {
            "timestamp": time.time(),
            "config": {
                "num_requests": 200,
                "warmup_requests": 20,
                "test_prompts": [
                    "The quick brown fox jumps over the lazy dog.",
                    "In a hole in the ground there lived a hobbit.",
                    "It was the best of times, it was the worst of times.",
                    "To be or not to be, that is the question.",
                    "All happy families are alike; each unhappy family is unhappy in its own way."
                ]
            },
            "results": {}
        }
        # Register and load the model for benchmarking
        model_id = "gpt2:1"
        self.inferneo_engine.model_manager.registered_models[model_id] = ModelMetadata(
            name="gpt2",
            version="1",
            path="models/huggingface/gpt2",
            format=ModelFormat.HUGGINGFACE,
            config={},
            metadata={},
            created_at=time.time()
        )
        # Also add to .models for active_models property
        self.inferneo_engine.model_manager.models[model_id] = self.inferneo_engine.model_manager.registered_models[model_id]
        asyncio.run(self.inferneo_engine.load_model("gpt2:1"))
        self.inferneo_engine.default_model = "gpt2:1"
        self.inferneo_engine.loaded_models["gpt2:1"] = self.inferneo_engine.model_manager.loaded_models["gpt2:1"]
    
    def generate_test_prompts(self, count: int) -> List[str]:
        """Generate test prompts"""
        base_prompts = self.results["config"]["test_prompts"]
        prompts = []
        for i in range(count):
            prompts.append(base_prompts[i % len(base_prompts)])
        return prompts
    
    def benchmark_inferneo_latency(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Inferneo latency"""
        logger.info("Benchmarking Inferneo latency...")
        
        # Warmup
        for _ in range(self.results["config"]["warmup_requests"]):
            asyncio.run(self.inferneo_engine.generate(prompts[0], max_tokens=50))
        
        # Actual benchmark
        latencies = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            request_start = time.time()
            try:
                response = asyncio.run(self.inferneo_engine.generate(prompt, max_tokens=50))
                latency = (time.time() - request_start) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Inferneo request {i} failed: {e}")
                latencies.append(float('inf'))
            
            if (i + 1) % 50 == 0:
                logger.info(f"Inferneo: Completed {i + 1}/{len(prompts)} requests")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        return {
            "count": len(valid_latencies),
            "mean": np.mean(valid_latencies),
            "median": np.median(valid_latencies),
            "min": np.min(valid_latencies),
            "max": np.max(valid_latencies),
            "std": np.std(valid_latencies),
            "p50": np.percentile(valid_latencies, 50),
            "p90": np.percentile(valid_latencies, 90),
            "p95": np.percentile(valid_latencies, 95),
            "p99": np.percentile(valid_latencies, 99),
            "errors": len(latencies) - len(valid_latencies),
            "total_time": total_time,
            "throughput": len(valid_latencies) / total_time
        }
    
    def benchmark_triton_latency(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Triton latency"""
        logger.info("Benchmarking Triton latency...")
        
        # Warmup
        for _ in range(self.results["config"]["warmup_requests"]):
            self._triton_request(prompts[0])
        
        # Actual benchmark
        latencies = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            request_start = time.time()
            try:
                response = self._triton_request(prompt)
                latency = (time.time() - request_start) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Triton request {i} failed: {e}")
                latencies.append(float('inf'))
            
            if (i + 1) % 50 == 0:
                logger.info(f"Triton: Completed {i + 1}/{len(prompts)} requests")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        return {
            "count": len(valid_latencies),
            "mean": np.mean(valid_latencies),
            "median": np.median(valid_latencies),
            "min": np.min(valid_latencies),
            "max": np.max(valid_latencies),
            "std": np.std(valid_latencies),
            "p50": np.percentile(valid_latencies, 50),
            "p90": np.percentile(valid_latencies, 90),
            "p95": np.percentile(valid_latencies, 95),
            "p99": np.percentile(valid_latencies, 99),
            "errors": len(latencies) - len(valid_latencies),
            "total_time": total_time,
            "throughput": len(valid_latencies) / total_time
        }
    
    def _triton_request(self, prompt: str) -> Dict[str, Any]:
        """Make a request to Triton server"""
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [prompt]
                }
            ],
            "outputs": [
                {
                    "name": "text_output"
                }
            ]
        }
        
        response = requests.post(
            f"{self.triton_url}/v2/models/gpt2/infer",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            outputs = result.get("outputs", [])
            if outputs:
                generated_text = outputs[0].get("data", [""])[0]
                return {
                    "text": generated_text,
                    "success": True
                }
        else:
            # Add debug information
            print(f"Triton request failed with status {response.status_code}")
            print(f"Response: {response.text}")
        
        raise Exception(f"Triton request failed: {response.status_code}")
    
    def benchmark_concurrent_requests(self, prompts: List[str], concurrency_levels: List[int]):
        """Benchmark concurrent request handling"""
        logger.info("Benchmarking concurrent requests...")
        
        results = {}
        
        for num_clients in concurrency_levels:
            logger.info(f"Testing {num_clients} concurrent clients")
            
            # Test Inferneo
            inferneo_result = self._benchmark_inferneo_concurrent(prompts, num_clients)
            
            # Test Triton
            triton_result = self._benchmark_triton_concurrent(prompts, num_clients)
            
            results[f"concurrency_{num_clients}"] = {
                "inferneo": inferneo_result,
                "triton": triton_result
            }
        
        return results
    
    def _benchmark_inferneo_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
        """Benchmark Inferneo concurrent requests"""
        latencies = []
        
        async def client_request(client_id: int):
            client_prompts = prompts[client_id::num_clients]
            client_latencies = []
            
            for prompt in client_prompts:
                start_time = time.time()
                try:
                    response = await self.inferneo_engine.generate(prompt, max_tokens=50)
                    latency = (time.time() - start_time) * 1000
                    client_latencies.append(latency)
                except Exception as e:
                    client_latencies.append(float('inf'))
            
            return client_latencies
        
        # Create concurrent tasks with proper event loop handling
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = [client_request(i) for i in range(num_clients)]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()
        except Exception as e:
            # Fallback to sequential execution
            results = []
            for i in range(num_clients):
                client_prompts = prompts[i::num_clients]
                client_latencies = []
                for prompt in client_prompts:
                    start_time = time.time()
                    try:
                        response = asyncio.run(self.inferneo_engine.generate(prompt, max_tokens=50))
                        latency = (time.time() - start_time) * 1000
                        client_latencies.append(latency)
                    except Exception:
                        client_latencies.append(float('inf'))
                results.append(client_latencies)
        
        # Combine all latencies
        for client_latencies in results:
            latencies.extend(client_latencies)
        
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        return {
            "count": len(valid_latencies),
            "mean": np.mean(valid_latencies),
            "p95": np.percentile(valid_latencies, 95),
            "p99": np.percentile(valid_latencies, 99),
            "errors": len(latencies) - len(valid_latencies),
            "throughput": len(valid_latencies) / (max(valid_latencies) / 1000) if valid_latencies else 0
        }
    
    def _benchmark_triton_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
        """Benchmark Triton concurrent requests"""
        latencies = []
        
        async def client_request(client_id: int):
            client_prompts = prompts[client_id::num_clients]
            client_latencies = []
            
            for prompt in client_prompts:
                start_time = time.time()
                try:
                    response = self._triton_request(prompt)
                    latency = (time.time() - start_time) * 1000
                    client_latencies.append(latency)
                except Exception as e:
                    client_latencies.append(float('inf'))
            
            return client_latencies
        
        # Create concurrent tasks with proper event loop handling
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tasks = [client_request(i) for i in range(num_clients)]
            results = loop.run_until_complete(asyncio.gather(*tasks))
            loop.close()
        except Exception as e:
            # Fallback to sequential execution
            results = []
            for i in range(num_clients):
                client_prompts = prompts[i::num_clients]
                client_latencies = []
                for prompt in client_prompts:
                    start_time = time.time()
                    try:
                        response = self._triton_request(prompt)
                        latency = (time.time() - start_time) * 1000
                        client_latencies.append(latency)
                    except Exception:
                        client_latencies.append(float('inf'))
                results.append(client_latencies)
        
        # Combine all latencies
        for client_latencies in results:
            latencies.extend(client_latencies)
        
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        return {
            "count": len(valid_latencies),
            "mean": np.mean(valid_latencies),
            "p95": np.percentile(valid_latencies, 95),
            "p99": np.percentile(valid_latencies, 99),
            "errors": len(latencies) - len(valid_latencies),
            "throughput": len(valid_latencies) / (max(valid_latencies) / 1000) if valid_latencies else 0
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark: Inferneo vs Triton")
        
        # Generate test prompts
        prompts = self.generate_test_prompts(self.results["config"]["num_requests"])
        
        # 1. Latency Benchmark
        logger.info("=" * 60)
        logger.info("LATENCY BENCHMARK")
        logger.info("=" * 60)
        
        inferneo_latency = self.benchmark_inferneo_latency(prompts)
        triton_latency = self.benchmark_triton_latency(prompts)
        
        self.results["results"]["latency"] = {
            "inferneo": inferneo_latency,
            "triton": triton_latency
        }
        
        # 2. Concurrency Benchmark
        logger.info("=" * 60)
        logger.info("CONCURRENCY BENCHMARK")
        logger.info("=" * 60)
        
        concurrency_levels = [1, 4, 8, 16]
        concurrency_results = self.benchmark_concurrent_requests(prompts, concurrency_levels)
        self.results["results"]["concurrency"] = concurrency_results
        
        # Generate comparison summary
        self.results["summary"] = self._generate_comparison_summary()
        
        return self.results
    
    def _generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate comparison summary"""
        latency_results = self.results["results"]["latency"]
        
        inferneo_latency = latency_results["inferneo"]
        triton_latency = latency_results["triton"]
        
        # Calculate performance ratios
        latency_ratio = triton_latency["mean"] / inferneo_latency["mean"] if inferneo_latency["mean"] > 0 else 0
        throughput_ratio = inferneo_latency["throughput"] / triton_latency["throughput"] if triton_latency["throughput"] > 0 else 0
        
        return {
            "latency_comparison": {
                "inferneo_mean_ms": inferneo_latency["mean"],
                "triton_mean_ms": triton_latency["mean"],
                "inferneo_p99_ms": inferneo_latency["p99"],
                "triton_p99_ms": triton_latency["p99"],
                "latency_ratio": latency_ratio,
                "inferneo_faster": latency_ratio > 1
            },
            "throughput_comparison": {
                "inferneo_rps": inferneo_latency["throughput"],
                "triton_rps": triton_latency["throughput"],
                "throughput_ratio": throughput_ratio,
                "inferneo_higher_throughput": throughput_ratio > 1
            },
            "reliability_comparison": {
                "inferneo_errors": inferneo_latency["errors"],
                "triton_errors": triton_latency["errors"],
                "inferneo_success_rate": (inferneo_latency["count"] / self.results["config"]["num_requests"]) * 100,
                "triton_success_rate": (triton_latency["count"] / self.results["config"]["num_requests"]) * 100
            }
        }
    
    def print_results(self):
        """Print benchmark results"""
        summary = self.results["summary"]
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK RESULTS: INFERNEO vs TRITON")
        print("=" * 80)
        
        # Latency Comparison
        print("\n📊 LATENCY COMPARISON:")
        print(f"  Inferneo Mean Latency: {summary['latency_comparison']['inferneo_mean_ms']:.2f} ms")
        print(f"  Triton Mean Latency:   {summary['latency_comparison']['triton_mean_ms']:.2f} ms")
        print(f"  Inferneo P99 Latency:  {summary['latency_comparison']['inferneo_p99_ms']:.2f} ms")
        print(f"  Triton P99 Latency:    {summary['latency_comparison']['triton_p99_ms']:.2f} ms")
        
        if summary['latency_comparison']['inferneo_faster']:
            print(f"  🏆 Inferneo is {summary['latency_comparison']['latency_ratio']:.2f}x FASTER")
        else:
            print(f"  🏆 Triton is {1/summary['latency_comparison']['latency_ratio']:.2f}x FASTER")
        
        # Throughput Comparison
        print("\n🚀 THROUGHPUT COMPARISON:")
        print(f"  Inferneo Throughput: {summary['throughput_comparison']['inferneo_rps']:.2f} RPS")
        print(f"  Triton Throughput:   {summary['throughput_comparison']['triton_rps']:.2f} RPS")
        
        if summary['throughput_comparison']['inferneo_higher_throughput']:
            print(f"  🏆 Inferneo has {summary['throughput_comparison']['throughput_ratio']:.2f}x HIGHER THROUGHPUT")
        else:
            print(f"  🏆 Triton has {1/summary['throughput_comparison']['throughput_ratio']:.2f}x HIGHER THROUGHPUT")
        
        # Reliability Comparison
        print("\n🛡️ RELIABILITY COMPARISON:")
        print(f"  Inferneo Success Rate: {summary['reliability_comparison']['inferneo_success_rate']:.1f}%")
        print(f"  Triton Success Rate:   {summary['reliability_comparison']['triton_success_rate']:.1f}%")
        print(f"  Inferneo Errors: {summary['reliability_comparison']['inferneo_errors']}")
        print(f"  Triton Errors:   {summary['reliability_comparison']['triton_errors']}")
        
        # Overall Winner
        print("\n🏁 OVERALL ASSESSMENT:")
        inferneo_score = 0
        triton_score = 0
        
        if summary['latency_comparison']['inferneo_faster']:
            inferneo_score += 1
        else:
            triton_score += 1
        
        if summary['throughput_comparison']['inferneo_higher_throughput']:
            inferneo_score += 1
        else:
            triton_score += 1
        
        if summary['reliability_comparison']['inferneo_success_rate'] > summary['reliability_comparison']['triton_success_rate']:
            inferneo_score += 1
        else:
            triton_score += 1
        
        if inferneo_score > triton_score:
            print("  🏆 INFERNEO WINS - Better overall performance")
        elif triton_score > inferneo_score:
            print("  🏆 TRITON WINS - Better overall performance")
        else:
            print("  🤝 TIE - Comparable performance")
        
        print(f"  Score: Inferneo {inferneo_score} - Triton {triton_score}")
        print("=" * 80)


def main():
    """Main benchmark function"""
    benchmark = ComprehensiveBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_results()
        
        # Save results
        timestamp = int(time.time())
        report_path = f"comprehensive_comparison_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main() 