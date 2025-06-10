#!/usr/bin/env python3
"""
Production Benchmark: Inferneo vs NVIDIA Triton

Comprehensive performance comparison with industry-standard parameters.
"""

import asyncio
import time
import statistics
import json
import subprocess
import requests
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import argparse
import logging
from pathlib import Path
import sys
sys.path.append('.')
import os

# Import Triton client
from benchmarks.triton_client import triton_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace any hardcoded Hugging Face token with environment variable
HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN_HERE')  # Set this in your .env file or environment

class BenchmarkType(Enum):
    SINGLE_REQUEST = "single_request"
    BATCH_PROCESSING = "batch_processing"
    CONCURRENT_REQUESTS = "concurrent_requests"
    THROUGHPUT = "throughput"
    LATENCY_DISTRIBUTION = "latency_distribution"

@dataclass
class BenchmarkConfig:
    """Configuration for production benchmarks"""
    # Model configurations
    model_name: str = "gpt2"
    model_size: str = "small"  # small, medium, large
    
    # Request configurations
    prompt_length: int = 50
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Benchmark configurations
    num_requests: int = 1000
    batch_sizes: List[int] = None
    concurrent_clients: List[int] = None
    warmup_requests: int = 50
    
    # System configurations
    gpu_memory_fraction: float = 0.8
    max_batch_size: int = 32
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32]
        if self.concurrent_clients is None:
            self.concurrent_clients = [1, 4, 8, 16, 32]

class ProductionBenchmark:
    """Production benchmark comparing Inferneo vs Triton"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.triton_url = "http://localhost:8003"
        self.inferneo_engine = None
        self.triton_available = True
        
    async def setup(self):
        """Setup benchmark environment"""
        logger.info("Setting up production benchmark environment...")
        
        # Download model if needed
        await self._download_model()
        
        # Setup Inferneo
        await self._setup_inferneo()
        
        # Setup Triton (verify it's running)
        await self._verify_triton()
        
        logger.info("Benchmark environment ready")
    
    async def _download_model(self):
        """Download model from HuggingFace"""
        logger.info(f"Downloading model: {self.config.model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Set token
            os.environ["HF_TOKEN"] = HF_TOKEN
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                token=HF_TOKEN,
                cache_dir="./models/huggingface"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                token=HF_TOKEN,
                cache_dir="./models/huggingface",
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Save locally for Triton
            model_path = f"./models/huggingface/{self.config.model_name}"
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            logger.info(f"Model downloaded to {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    async def _setup_inferneo(self):
        """Setup Inferneo engine"""
        logger.info("Setting up Inferneo engine...")
        
        try:
            from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig
            from inferneo.models.base import ModelFormat
            
            config = EnhancedEngineConfig(
                model=None,
                max_workers=8,
                max_memory_gb=16,
                max_concurrent_models=3,
                enable_ab_testing=False,
                enable_performance_monitoring=True,
                enable_health_checks=False,
                metrics_export_interval=60,
                max_waiting_tokens=4096,
                block_size=128
            )
            
            self.inferneo_engine = EnhancedInferneoEngine(config)
            await self.inferneo_engine.start()
            
            # Register the model
            await self.inferneo_engine.register_model(
                name=self.config.model_name,
                version="1.0",
                path=f"./models/huggingface/{self.config.model_name}",
                format=ModelFormat.HUGGINGFACE
            )
            
            logger.info("Inferneo engine ready")
            
        except Exception as e:
            logger.error(f"Failed to setup Inferneo: {e}")
            raise
    
    async def _verify_triton(self):
        """Verify Triton server is running"""
        logger.info("Verifying Triton server...")
        
        try:
            is_healthy = await triton_client.health_check()
            if is_healthy:
                logger.info("Triton server is ready")
                
                # List available models
                models = await triton_client.list_models()
                logger.info(f"Available Triton models: {models}")
                
                if not models:
                    logger.warning("No models found in Triton server - Triton benchmarks will be skipped")
                    self.triton_available = False
                else:
                    self.triton_available = True
            else:
                logger.warning("Triton server is not healthy - Triton benchmarks will be skipped")
                self.triton_available = False
        except Exception as e:
            logger.warning(f"Could not connect to Triton server: {e}")
            logger.info("Triton benchmarks will be skipped")
            self.triton_available = False
    
    async def run_benchmarks(self):
        """Run all benchmarks"""
        logger.info("Starting production benchmarks...")
        
        # Generate test prompts
        prompts = self._generate_test_prompts()
        
        # Run different benchmark types
        benchmark_types = [
            BenchmarkType.SINGLE_REQUEST,
            BenchmarkType.BATCH_PROCESSING,
            BenchmarkType.CONCURRENT_REQUESTS,
            BenchmarkType.THROUGHPUT,
            BenchmarkType.LATENCY_DISTRIBUTION
        ]
        
        for benchmark_type in benchmark_types:
            logger.info(f"Running {benchmark_type.value} benchmark...")
            
            if benchmark_type == BenchmarkType.SINGLE_REQUEST:
                await self._benchmark_single_requests(prompts)
            elif benchmark_type == BenchmarkType.BATCH_PROCESSING:
                await self._benchmark_batch_processing(prompts)
            elif benchmark_type == BenchmarkType.CONCURRENT_REQUESTS:
                await self._benchmark_concurrent_requests(prompts)
            elif benchmark_type == BenchmarkType.THROUGHPUT:
                await self._benchmark_throughput(prompts)
            elif benchmark_type == BenchmarkType.LATENCY_DISTRIBUTION:
                await self._benchmark_latency_distribution(prompts)
        
        # Generate comprehensive report
        await self._generate_report()
    
    def _generate_test_prompts(self) -> List[str]:
        """Generate realistic test prompts"""
        base_prompts = [
            "The future of artificial intelligence lies in",
            "Machine learning models have revolutionized",
            "In the context of natural language processing,",
            "The development of large language models",
            "Artificial intelligence applications in healthcare",
            "The impact of deep learning on computer vision",
            "Neural networks have transformed the way we",
            "The evolution of transformer architectures",
            "Quantum computing applications in AI",
            "The role of reinforcement learning in"
        ]
        
        prompts = []
        for i in range(self.config.num_requests):
            base = base_prompts[i % len(base_prompts)]
            # Add some variation
            variation = f" (scenario {i+1})" if i < len(base_prompts) else ""
            prompts.append(base + variation)
        
        return prompts
    
    async def _benchmark_single_requests(self, prompts: List[str]):
        """Benchmark single request latency"""
        logger.info("Benchmarking single request latency...")
        
        results = {
            "inferneo": await self._benchmark_inferneo_single(prompts)
        }
        
        if self.triton_available:
            results["triton"] = await self._benchmark_triton_single(prompts)
        
        self.results["single_requests"] = results
        self._print_single_request_results(results)
    
    async def _benchmark_inferneo_single(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Inferneo single requests"""
        latencies = []
        
        # Warmup
        for _ in range(self.config.warmup_requests):
            try:
                await self.inferneo_engine.generate(prompts[0])
            except:
                pass
        
        # Actual benchmark
        for i, prompt in enumerate(prompts[:self.config.num_requests]):
            start_time = time.time()
            try:
                result = await self.inferneo_engine.generate(prompt)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Inferneo request {i} failed: {e}")
                latencies.append(float('inf'))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Inferneo: Completed {i + 1}/{self.config.num_requests} requests")
        
        return self._calculate_statistics(latencies)
    
    async def _benchmark_triton_single(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Triton single requests"""
        latencies = []
        
        # Warmup
        for _ in range(self.config.warmup_requests):
            try:
                await self._triton_request(prompts[0])
            except:
                pass
        
        # Actual benchmark
        for i, prompt in enumerate(prompts[:self.config.num_requests]):
            start_time = time.time()
            try:
                result = await self._triton_request(prompt)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Triton request {i} failed: {e}")
                latencies.append(float('inf'))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Triton: Completed {i + 1}/{self.config.num_requests} requests")
        
        return self._calculate_statistics(latencies)
    
    async def _triton_request(self, prompt: str) -> Dict[str, Any]:
        """Make a request to Triton server"""
        return await triton_client.generate_text(
            prompt, 
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
    
    async def _benchmark_batch_processing(self, prompts: List[str]):
        """Benchmark batch processing performance"""
        logger.info("Benchmarking batch processing...")
        
        results = {}
        
        for batch_size in self.config.batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            batch_prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
            
            results[f"batch_size_{batch_size}"] = {
                "inferneo": await self._benchmark_inferneo_batch(batch_prompts)
            }
            
            if self.triton_available:
                results[f"batch_size_{batch_size}"]["triton"] = await self._benchmark_triton_batch(batch_prompts)
        
        self.results["batch_processing"] = results
        self._print_batch_results(results)
    
    async def _benchmark_inferneo_batch(self, batch_prompts: List[List[str]]) -> Dict[str, Any]:
        """Benchmark Inferneo batch processing"""
        latencies = []
        
        for batch in batch_prompts:
            start_time = time.time()
            try:
                results = await self.inferneo_engine.generate_batch(batch)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Inferneo batch failed: {e}")
                latencies.append(float('inf'))
        
        return self._calculate_statistics(latencies)
    
    async def _benchmark_triton_batch(self, batch_prompts: List[List[str]]) -> Dict[str, Any]:
        """Benchmark Triton batch processing"""
        latencies = []
        
        for batch in batch_prompts:
            start_time = time.time()
            try:
                results = await self._triton_batch_request(batch)
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Triton batch failed: {e}")
                latencies.append(float('inf'))
        
        return self._calculate_statistics(latencies)
    
    async def _triton_batch_request(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Make a batch request to Triton server"""
        return await triton_client.generate_batch(
            prompts,
            model_name=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
    
    async def _benchmark_concurrent_requests(self, prompts: List[str]):
        """Benchmark concurrent request handling"""
        logger.info("Benchmarking concurrent requests...")
        
        results = {}
        
        for num_clients in self.config.concurrent_clients:
            logger.info(f"Testing {num_clients} concurrent clients")
            
            results[f"clients_{num_clients}"] = {
                "inferneo": await self._benchmark_inferneo_concurrent(prompts, num_clients)
            }
            
            if self.triton_available:
                results[f"clients_{num_clients}"]["triton"] = await self._benchmark_triton_concurrent(prompts, num_clients)
        
        self.results["concurrent_requests"] = results
        self._print_concurrent_results(results)
    
    async def _benchmark_inferneo_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
        """Benchmark Inferneo concurrent requests"""
        latencies = []
        
        async def client_request(client_id: int):
            client_prompts = prompts[client_id::num_clients]
            client_latencies = []
            
            for prompt in client_prompts:
                start_time = time.time()
                try:
                    result = await self.inferneo_engine.generate(prompt)
                    latency = (time.time() - start_time) * 1000
                    client_latencies.append(latency)
                except Exception as e:
                    client_latencies.append(float('inf'))
            
            return client_latencies
        
        # Create concurrent tasks
        tasks = [client_request(i) for i in range(num_clients)]
        results = await asyncio.gather(*tasks)
        
        # Combine all latencies
        for client_latencies in results:
            latencies.extend(client_latencies)
        
        return self._calculate_statistics(latencies)
    
    async def _benchmark_triton_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
        """Benchmark Triton concurrent requests"""
        latencies = []
        
        async def client_request(client_id: int):
            client_prompts = prompts[client_id::num_clients]
            client_latencies = []
            
            for prompt in client_prompts:
                start_time = time.time()
                try:
                    result = await self._triton_request(prompt)
                    latency = (time.time() - start_time) * 1000
                    client_latencies.append(latency)
                except Exception as e:
                    client_latencies.append(float('inf'))
            
            return client_latencies
        
        # Create concurrent tasks
        tasks = [client_request(i) for i in range(num_clients)]
        results = await asyncio.gather(*tasks)
        
        # Combine all latencies
        for client_latencies in results:
            latencies.extend(client_latencies)
        
        return self._calculate_statistics(latencies)
    
    async def _benchmark_throughput(self, prompts: List[str]):
        """Benchmark throughput (requests per second)"""
        logger.info("Benchmarking throughput...")
        
        # Test different request rates
        request_rates = [10, 50, 100, 200, 500]  # requests per second
        results = {}
        
        for rate in request_rates:
            logger.info(f"Testing throughput at {rate} req/s")
            
            results[f"rate_{rate}"] = {
                "inferneo": await self._benchmark_inferneo_throughput(prompts, rate)
            }
            
            if self.triton_available:
                results[f"rate_{rate}"]["triton"] = await self._benchmark_triton_throughput(prompts, rate)
        
        self.results["throughput"] = results
        self._print_throughput_results(results)
    
    async def _benchmark_inferneo_throughput(self, prompts: List[str], target_rate: int) -> Dict[str, Any]:
        """Benchmark Inferneo throughput"""
        interval = 1.0 / target_rate
        latencies = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            request_start = time.time()
            try:
                result = await self.inferneo_engine.generate(prompt)
                latency = (time.time() - request_start) * 1000
                latencies.append(latency)
            except Exception as e:
                latencies.append(float('inf'))
            
            # Rate limiting
            elapsed = time.time() - start_time
            expected_time = (i + 1) * interval
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)
        
        total_time = time.time() - start_time
        actual_rate = len(prompts) / total_time
        
        return {
            "target_rate": target_rate,
            "actual_rate": actual_rate,
            "latency_stats": self._calculate_statistics(latencies),
            "total_time": total_time
        }
    
    async def _benchmark_triton_throughput(self, prompts: List[str], target_rate: int) -> Dict[str, Any]:
        """Benchmark Triton throughput"""
        interval = 1.0 / target_rate
        latencies = []
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            request_start = time.time()
            try:
                result = await self._triton_request(prompt)
                latency = (time.time() - request_start) * 1000
                latencies.append(latency)
            except Exception as e:
                latencies.append(float('inf'))
            
            # Rate limiting
            elapsed = time.time() - start_time
            expected_time = (i + 1) * interval
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)
        
        total_time = time.time() - start_time
        actual_rate = len(prompts) / total_time
        
        return {
            "target_rate": target_rate,
            "actual_rate": actual_rate,
            "latency_stats": self._calculate_statistics(latencies),
            "total_time": total_time
        }
    
    async def _benchmark_latency_distribution(self, prompts: List[str]):
        """Benchmark latency distribution analysis"""
        logger.info("Benchmarking latency distribution...")
        
        # Run a large number of requests to get good distribution
        num_requests = min(1000, len(prompts))
        test_prompts = prompts[:num_requests]
        
        results = {
            "inferneo": await self._benchmark_inferneo_latency_distribution(test_prompts)
        }
        
        if self.triton_available:
            results["triton"] = await self._benchmark_triton_latency_distribution(test_prompts)
        
        self.results["latency_distribution"] = results
        self._print_latency_distribution_results(results)
    
    async def _benchmark_inferneo_latency_distribution(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Inferneo latency distribution"""
        latencies = []
        
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            try:
                result = await self.inferneo_engine.generate(prompt)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                latencies.append(float('inf'))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Inferneo distribution: {i + 1}/{len(prompts)}")
        
        return self._calculate_detailed_statistics(latencies)
    
    async def _benchmark_triton_latency_distribution(self, prompts: List[str]) -> Dict[str, Any]:
        """Benchmark Triton latency distribution"""
        latencies = []
        
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            try:
                result = await self._triton_request(prompt)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            except Exception as e:
                latencies.append(float('inf'))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Triton distribution: {i + 1}/{len(prompts)}")
        
        return self._calculate_detailed_statistics(latencies)
    
    def _calculate_statistics(self, latencies: List[float]) -> Dict[str, Any]:
        """Calculate basic statistics from latency data"""
        valid_latencies = [l for l in latencies if l != float('inf')]
        
        if not valid_latencies:
            return {"error": "No valid latencies"}
        
        return {
            "count": len(valid_latencies),
            "mean": statistics.mean(valid_latencies),
            "median": statistics.median(valid_latencies),
            "min": min(valid_latencies),
            "max": max(valid_latencies),
            "std": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
            "p50": np.percentile(valid_latencies, 50),
            "p90": np.percentile(valid_latencies, 90),
            "p95": np.percentile(valid_latencies, 95),
            "p99": np.percentile(valid_latencies, 99),
            "errors": len(latencies) - len(valid_latencies)
        }
    
    def _calculate_detailed_statistics(self, latencies: List[float]) -> Dict[str, Any]:
        """Calculate detailed statistics including percentiles"""
        basic_stats = self._calculate_statistics(latencies)
        
        valid_latencies = [l for l in latencies if l != float('inf')]
        if not valid_latencies:
            return basic_stats
        
        # Additional percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        percentile_values = {f"p{p}": np.percentile(valid_latencies, p) for p in percentiles}
        
        basic_stats.update(percentile_values)
        return basic_stats
    
    def _print_single_request_results(self, results: Dict[str, Any]):
        """Print single request benchmark results"""
        print("\n" + "="*60)
        print("SINGLE REQUEST LATENCY BENCHMARK")
        print("="*60)
        
        for system, stats in results.items():
            print(f"\n{system.upper()}:")
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                print(f"  Mean: {stats['mean']:.2f} ms")
                print(f"  Median: {stats['median']:.2f} ms")
                print(f"  P95: {stats['p95']:.2f} ms")
                print(f"  P99: {stats['p99']:.2f} ms")
                print(f"  Std Dev: {stats['std']:.2f} ms")
                print(f"  Errors: {stats['errors']}")
    
    def _print_batch_results(self, results: Dict[str, Any]):
        """Print batch processing benchmark results"""
        print("\n" + "="*60)
        print("BATCH PROCESSING BENCHMARK")
        print("="*60)
        
        for batch_size, systems in results.items():
            print(f"\nBatch Size: {batch_size}")
            # Extract integer batch size
            try:
                batch_size_int = int(batch_size.split('_')[-1])
            except Exception:
                batch_size_int = 1
            for system, stats in systems.items():
                print(f"  {system.upper()}:")
                if "error" in stats:
                    print(f"    Error: {stats['error']}")
                else:
                    print(f"    Mean: {stats['mean']:.2f} ms")
                    print(f"    Throughput: {batch_size_int * 1000 / stats['mean']:.1f} tokens/s")
    
    def _print_concurrent_results(self, results: Dict[str, Any]):
        """Print concurrent request benchmark results"""
        print("\n" + "="*60)
        print("CONCURRENT REQUESTS BENCHMARK")
        print("="*60)
        
        for clients, systems in results.items():
            print(f"\nConcurrent Clients: {clients}")
            for system, stats in systems.items():
                print(f"  {system.upper()}:")
                if "error" in stats:
                    print(f"    Error: {stats['error']}")
                else:
                    print(f"    Mean: {stats['mean']:.2f} ms")
                    print(f"    Throughput: {stats['count'] * 1000 / stats['mean']:.1f} req/s")
    
    def _print_throughput_results(self, results: Dict[str, Any]):
        """Print throughput benchmark results"""
        print("\n" + "="*60)
        print("THROUGHPUT BENCHMARK")
        print("="*60)
        
        for rate, systems in results.items():
            print(f"\nTarget Rate: {rate} req/s")
            for system, stats in systems.items():
                print(f"  {system.upper()}:")
                print(f"    Actual Rate: {stats['actual_rate']:.1f} req/s")
                print(f"    Mean Latency: {stats['latency_stats']['mean']:.2f} ms")
    
    def _print_latency_distribution_results(self, results: Dict[str, Any]):
        """Print latency distribution benchmark results"""
        print("\n" + "="*60)
        print("LATENCY DISTRIBUTION ANALYSIS")
        print("="*60)
        
        for system, stats in results.items():
            print(f"\n{system.upper()}:")
            if "error" in stats:
                print(f"  Error: {stats['error']}")
            else:
                print(f"  P50: {stats['p50']:.2f} ms")
                print(f"  P90: {stats['p90']:.2f} ms")
                print(f"  P95: {stats['p95']:.2f} ms")
                print(f"  P99: {stats['p99']:.2f} ms")
                print(f"  P99.9: {stats['p99.9']:.2f} ms")
    
    async def _generate_report(self):
        """Generate comprehensive benchmark report"""
        logger.info("Generating comprehensive benchmark report...")
        
        report = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        # Save report
        report_path = f"production_benchmark_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        
        # Print summary
        self._print_summary(report["summary"])
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmark summary"""
        summary = {
            "model": self.config.model_name,
            "total_requests": self.config.num_requests,
            "systems_tested": list(self.results["single_requests"].keys()),
            "recommendations": []
        }
        # Analyze results and provide recommendations
        if "single_requests" in self.results:
            inferneo_latency = self.results["single_requests"].get("inferneo", {}).get("mean", float('inf'))
            triton_latency = self.results["single_requests"].get("triton", {}).get("mean", float('inf'))
            if inferneo_latency != float('inf') and triton_latency != float('inf'):
                if inferneo_latency < triton_latency:
                    summary["recommendations"].append("Inferneo shows better single-request latency")
                else:
                    summary["recommendations"].append("Triton shows better single-request latency")
            elif inferneo_latency != float('inf'):
                summary["recommendations"].append("Only Inferneo results available")
            elif triton_latency != float('inf'):
                summary["recommendations"].append("Only Triton results available")
            else:
                summary["recommendations"].append("No valid latency results available")
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print(f"Model: {summary['model']}")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Systems Tested: {', '.join(summary['systems_tested'])}")
        print("\nRecommendations:")
        for rec in summary["recommendations"]:
            print(f"  - {rec}")
    
    async def cleanup(self):
        """Cleanup benchmark environment"""
        logger.info("Cleaning up benchmark environment...")
        
        if self.inferneo_engine:
            await self.inferneo_engine.stop()

async def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="Production Benchmark: Inferneo vs Triton")
    parser.add_argument("--model", default="gpt2", help="Model to benchmark")
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32], help="Batch sizes to test")
    parser.add_argument("--concurrent-clients", nargs="+", type=int, default=[1, 4, 8, 16, 32], help="Concurrent clients to test")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup requests")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        model_name=args.model,
        num_requests=args.num_requests,
        batch_sizes=args.batch_sizes,
        concurrent_clients=args.concurrent_clients,
        warmup_requests=args.warmup
    )
    
    # Create and run benchmark
    benchmark = ProductionBenchmark(config)
    
    try:
        await benchmark.setup()
        await benchmark.run_benchmarks()
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 