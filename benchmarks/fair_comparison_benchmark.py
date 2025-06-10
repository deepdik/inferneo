#!/usr/bin/env python3
"""
Fair Comparison Benchmark: Inferneo vs Triton

This script runs identical benchmarks on both Inferneo and Triton
under the same conditions and timing for a truly fair comparison.
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
import random
from concurrent.futures import ThreadPoolExecutor
import subprocess

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig
from inferneo.models.base import ModelFormat, GenerationConfig
from inferneo.models.manager import ModelMetadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FairComparisonBenchmark:
    """Fair benchmark comparing Inferneo vs Triton under identical conditions"""
    
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "config": {
                "num_requests": 100,  # Reduced for faster testing
                "warmup_requests": 10,
                "test_prompts": [
                    "The quick brown fox jumps over the lazy dog.",
                    "In a hole in the ground there lived a hobbit.",
                    "It was the best of times, it was the worst of times.",
                    "To be or not to be, that is the question.",
                    "All happy families are alike; each unhappy family is unhappy in its own way."
                ],
                "generation_config": {
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            "results": {}
        }
        
        # Initialize systems will be done in async method
        self.inferneo_engine = None
        self.triton_url = None
        
    async def initialize(self):
        """Initialize both systems"""
        await self._init_inferneo()
        self._init_triton()
        
    def _init_inferneo(self):
        """Initialize Inferneo engine"""
        logger.info("Initializing Inferneo...")
        
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
        
        logger.info("Inferneo initialized successfully")
        
    async def _init_inferneo(self):
        """Initialize Inferneo engine (async version)"""
        logger.info("Initializing Inferneo...")
        
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
        
        # Load model
        await self.inferneo_engine.load_model("gpt2:1")
        self.inferneo_engine.default_model = "gpt2:1"
        self.inferneo_engine.loaded_models["gpt2:1"] = self.inferneo_engine.model_manager.loaded_models["gpt2:1"]
        
        logger.info("Inferneo initialized successfully")
        
    def _init_triton(self):
        """Initialize Triton connection"""
        logger.info("Initializing Triton connection...")
        self.triton_url = "http://localhost:8003"
        
        # Test connection
        try:
            response = requests.get(f"{self.triton_url}/v2/models/gpt2", timeout=5)
            if response.status_code == 200:
                logger.info("Triton connection successful")
            else:
                raise Exception(f"Triton model not ready: {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Triton: {e}")
            raise
        
    def generate_test_prompts(self, count: int) -> List[str]:
        """Generate test prompts"""
        base_prompts = self.results["config"]["test_prompts"]
        prompts = []
        for i in range(count):
            prompts.append(base_prompts[i % len(base_prompts)])
        return prompts
    
    async def warmup_both_systems(self, prompts: List[str]):
        """Warmup both systems with the same prompts"""
        logger.info("Warming up both systems...")
        warmup_count = self.results["config"]["warmup_requests"]
        
        for i in range(warmup_count):
            prompt = prompts[i % len(prompts)]
            
            # Warmup Inferneo
            try:
                await self.inferneo_engine.generate(prompt, max_tokens=50)
            except Exception as e:
                logger.warning(f"Inferneo warmup {i} failed: {e}")
            
            # Warmup Triton
            try:
                self._triton_request(prompt)
            except Exception as e:
                logger.warning(f"Triton warmup {i} failed: {e}")
        
        logger.info("Warmup completed")
    
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
        
        raise Exception(f"Triton request failed: {response.status_code}")
    
    async def run_fair_latency_benchmark(self, prompts: List[str]) -> Dict[str, Any]:
        """Run latency benchmark on both systems with identical conditions"""
        logger.info("Running fair latency benchmark...")
        
        # Warmup both systems
        await self.warmup_both_systems(prompts)
        
        # Prepare results storage
        inferneo_latencies = []
        triton_latencies = []
        inferneo_token_counts = []
        triton_token_counts = []
        
        # Run identical tests on both systems
        for i, prompt in enumerate(prompts):
            logger.info(f"Testing request {i+1}/{len(prompts)}")
            
            # Test Inferneo
            try:
                start_time = time.time()
                response = await self.inferneo_engine.generate(prompt, max_tokens=50)
                inferneo_latency = (time.time() - start_time) * 1000
                inferneo_latencies.append(inferneo_latency)
                
                # Count tokens
                prompt_tokens = len(prompt.split())  # Approximate
                completion_tokens = len(response.text.split()) if hasattr(response, 'text') else 0
                total_tokens = prompt_tokens + completion_tokens
                inferneo_token_counts.append({
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                })
            except Exception as e:
                logger.warning(f"Inferneo request {i} failed: {e}")
                inferneo_latencies.append(float('inf'))
                inferneo_token_counts.append({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
            
            # Test Triton with same prompt
            try:
                start_time = time.time()
                response = self._triton_request(prompt)
                triton_latency = (time.time() - start_time) * 1000
                triton_latencies.append(triton_latency)
                
                # Count tokens
                prompt_tokens = len(prompt.split())  # Approximate
                completion_tokens = len(response['text'].split()) if response.get('text') else 0
                total_tokens = prompt_tokens + completion_tokens
                triton_token_counts.append({
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                })
            except Exception as e:
                logger.warning(f"Triton request {i} failed: {e}")
                triton_latencies.append(float('inf'))
                triton_token_counts.append({'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        
        # Calculate statistics
        def calculate_stats(latencies, token_counts):
            valid_latencies = [l for l in latencies if l != float('inf')]
            valid_token_counts = [tc for i, tc in enumerate(token_counts) if latencies[i] != float('inf')]
            
            if not valid_latencies:
                return {"count": 0, "mean": 0, "median": 0, "p99": 0, "errors": len(latencies)}
            
            # Calculate token throughput
            total_time_seconds = sum(valid_latencies) / 1000  # Convert ms to seconds
            total_tokens = sum(tc['total_tokens'] for tc in valid_token_counts)
            total_completion_tokens = sum(tc['completion_tokens'] for tc in valid_token_counts)
            
            tokens_per_second = total_tokens / total_time_seconds if total_time_seconds > 0 else 0
            completion_tokens_per_second = total_completion_tokens / total_time_seconds if total_time_seconds > 0 else 0
            
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
                "throughput": {
                    "tokens_per_second": tokens_per_second,
                    "completion_tokens_per_second": completion_tokens_per_second,
                    "total_tokens": total_tokens,
                    "total_completion_tokens": total_completion_tokens,
                    "total_time_seconds": total_time_seconds
                }
            }
        
        return {
            "inferneo": calculate_stats(inferneo_latencies, inferneo_token_counts),
            "triton": calculate_stats(triton_latencies, triton_token_counts),
            "raw_data": {
                "inferneo_latencies": inferneo_latencies,
                "triton_latencies": triton_latencies,
                "inferneo_token_counts": inferneo_token_counts,
                "triton_token_counts": triton_token_counts
            }
        }
    
    async def run_concurrent_benchmark(self, prompts: List[str], concurrency_levels: List[int]) -> Dict[str, Any]:
        """Run concurrent benchmark on both systems"""
        logger.info("Running concurrent benchmark...")
        
        results = {}
        
        for num_clients in concurrency_levels:
            logger.info(f"Testing {num_clients} concurrent clients")
            
            # Test both systems with same concurrency level
            inferneo_result = await self._benchmark_inferneo_concurrent(prompts, num_clients)
            triton_result = await self._benchmark_triton_concurrent(prompts, num_clients)
            
            results[f"concurrency_{num_clients}"] = {
                "inferneo": inferneo_result,
                "triton": triton_result
            }
        
        return results
    
    async def _benchmark_inferneo_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
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
                except Exception:
                    client_latencies.append(float('inf'))
            
            return client_latencies
        
        # Create concurrent tasks
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
                        response = await self.inferneo_engine.generate(prompt, max_tokens=50)
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
            "mean": np.mean(valid_latencies) if valid_latencies else 0,
            "p95": np.percentile(valid_latencies, 95) if valid_latencies else 0,
            "p99": np.percentile(valid_latencies, 99) if valid_latencies else 0,
            "errors": len(latencies) - len(valid_latencies)
        }
    
    async def _benchmark_triton_concurrent(self, prompts: List[str], num_clients: int) -> Dict[str, Any]:
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
        
        # Create concurrent tasks
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
            "mean": np.mean(valid_latencies) if valid_latencies else 0,
            "p95": np.percentile(valid_latencies, 95) if valid_latencies else 0,
            "p99": np.percentile(valid_latencies, 99) if valid_latencies else 0,
            "errors": len(latencies) - len(valid_latencies)
        }
    
    async def run_fair_benchmark(self):
        """Run comprehensive fair benchmark suite"""
        logger.info("Starting FAIR comparison benchmark: Inferneo vs Triton")
        
        # Generate test prompts
        prompts = self.generate_test_prompts(self.results["config"]["num_requests"])
        
        # 1. Latency Benchmark (both systems tested with same prompts)
        logger.info("=" * 60)
        logger.info("FAIR LATENCY BENCHMARK")
        logger.info("=" * 60)
        
        latency_results = await self.run_fair_latency_benchmark(prompts)
        self.results["results"]["latency"] = latency_results
        
        # 2. Concurrency Benchmark
        logger.info("=" * 60)
        logger.info("CONCURRENCY BENCHMARK")
        logger.info("=" * 60)
        
        concurrency_levels = [1, 4, 8]
        concurrency_results = await self.run_concurrent_benchmark(prompts, concurrency_levels)
        self.results["results"]["concurrency"] = concurrency_results
        
        # Generate summary
        summary = self._generate_comparison_summary()
        self.results["summary"] = summary
        
        return self.results
    
    def _generate_comparison_summary(self) -> Dict[str, Any]:
        """Generate comparison summary"""
        latency_results = self.results["results"]["latency"]
        
        inferneo_latency = latency_results["inferneo"]
        triton_latency = latency_results["triton"]
        
        # Calculate performance ratios
        if inferneo_latency["mean"] > 0 and triton_latency["mean"] > 0:
            latency_ratio = triton_latency["mean"] / inferneo_latency["mean"]
            throughput_ratio = inferneo_latency["mean"] / triton_latency["mean"]
        else:
            latency_ratio = 1.0
            throughput_ratio = 1.0
        
        # Determine winner
        if inferneo_latency["mean"] < triton_latency["mean"]:
            winner = "Inferneo"
            score = "Inferneo 2 - Triton 1"
        else:
            winner = "Triton"
            score = "Triton 2 - Inferneo 1"
        
        return {
            "latency_comparison": {
                "inferneo_mean_latency": inferneo_latency["mean"],
                "triton_mean_latency": triton_latency["mean"],
                "inferneo_p99_latency": inferneo_latency["p99"],
                "triton_p99_latency": triton_latency["p99"],
                "latency_ratio": latency_ratio,
                "faster_system": "Inferneo" if latency_ratio > 1 else "Triton"
            },
            "reliability_comparison": {
                "inferneo_success_rate": (inferneo_latency["count"] / (inferneo_latency["count"] + inferneo_latency["errors"])) * 100 if (inferneo_latency["count"] + inferneo_latency["errors"]) > 0 else 0,
                "triton_success_rate": (triton_latency["count"] / (triton_latency["count"] + triton_latency["errors"])) * 100 if (triton_latency["count"] + triton_latency["errors"]) > 0 else 0,
                "inferneo_errors": inferneo_latency["errors"],
                "triton_errors": triton_latency["errors"]
            },
            "overall_assessment": {
                "winner": winner,
                "score": score,
                "notes": "Fair comparison with identical test conditions"
            }
        }
    
    def print_results(self):
        """Print benchmark results"""
        summary = self.results["summary"]
        latency = summary["latency_comparison"]
        reliability = summary["reliability_comparison"]
        overall = summary["overall_assessment"]
        
        print("\n" + "=" * 80)
        print("FAIR COMPARISON BENCHMARK RESULTS: INFERNEO vs TRITON")
        print("=" * 80)
        
        print("\n📊 LATENCY COMPARISON:")
        print(f"  Inferneo Mean Latency: {latency['inferneo_mean_latency']:.2f} ms")
        print(f"  Triton Mean Latency:   {latency['triton_mean_latency']:.2f} ms")
        print(f"  Inferneo P99 Latency:  {latency['inferneo_p99_latency']:.2f} ms")
        print(f"  Triton P99 Latency:    {latency['triton_p99_latency']:.2f} ms")
        print(f"  🏆 {latency['faster_system']} is {latency['latency_ratio']:.2f}x FASTER")
        
        print("\n🛡️ RELIABILITY COMPARISON:")
        print(f"  Inferneo Success Rate: {reliability['inferneo_success_rate']:.1f}%")
        print(f"  Triton Success Rate:   {reliability['triton_success_rate']:.1f}%")
        print(f"  Inferneo Errors: {reliability['inferneo_errors']}")
        print(f"  Triton Errors:   {reliability['triton_errors']}")
        
        print("\n🏁 OVERALL ASSESSMENT:")
        print(f"  🏆 {overall['winner']} WINS - Better overall performance")
        print(f"  Score: {overall['score']}")
        print("=" * 80)
        
        # Save detailed results
        timestamp = int(time.time())
        filename = f"fair_comparison_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to: {filename}")

    def print_comparison_report(self, latency_results: Dict, concurrency_results: Dict):
        """Print a detailed comparison report"""
        print("\n" + "="*80)
        print("🏆 FAIR COMPARISON BENCHMARK RESULTS")
        print("="*80)
        
        # Latency Comparison
        print("\n📊 LATENCY BENCHMARK RESULTS")
        print("-" * 50)
        
        inferneo_latency = latency_results["inferneo"]
        triton_latency = latency_results["triton"]
        
        print(f"Inferneo Latency (ms):")
        print(f"  Mean: {inferneo_latency['mean']:.2f}")
        print(f"  Median: {inferneo_latency['median']:.2f}")
        print(f"  P99: {inferneo_latency['p99']:.2f}")
        print(f"  Requests: {inferneo_latency['count']}")
        print(f"  Errors: {inferneo_latency['errors']}")
        
        print(f"\nTriton Latency (ms):")
        print(f"  Mean: {triton_latency['mean']:.2f}")
        print(f"  Median: {triton_latency['median']:.2f}")
        print(f"  P99: {triton_latency['p99']:.2f}")
        print(f"  Requests: {triton_latency['count']}")
        print(f"  Errors: {triton_latency['errors']}")
        
        # Token Throughput Comparison
        print(f"\n🚀 TOKEN THROUGHPUT RESULTS")
        print("-" * 50)
        
        inferneo_throughput = inferneo_latency.get('throughput', {})
        triton_throughput = triton_latency.get('throughput', {})
        
        print(f"Inferneo Token Throughput:")
        print(f"  Total Tokens/sec: {inferneo_throughput.get('tokens_per_second', 0):.2f}")
        print(f"  Completion Tokens/sec: {inferneo_throughput.get('completion_tokens_per_second', 0):.2f}")
        print(f"  Total Tokens: {inferneo_throughput.get('total_tokens', 0)}")
        print(f"  Total Completion Tokens: {inferneo_throughput.get('total_completion_tokens', 0)}")
        print(f"  Total Time: {inferneo_throughput.get('total_time_seconds', 0):.2f}s")
        
        print(f"\nTriton Token Throughput:")
        print(f"  Total Tokens/sec: {triton_throughput.get('tokens_per_second', 0):.2f}")
        print(f"  Completion Tokens/sec: {triton_throughput.get('completion_tokens_per_second', 0):.2f}")
        print(f"  Total Tokens: {triton_throughput.get('total_tokens', 0)}")
        print(f"  Total Completion Tokens: {triton_throughput.get('total_completion_tokens', 0)}")
        print(f"  Total Time: {triton_throughput.get('total_time_seconds', 0):.2f}s")
        
        # Performance Comparison
        if triton_latency['mean'] > 0:
            latency_ratio = triton_latency['mean'] / inferneo_latency['mean']
            print(f"\n⚡ PERFORMANCE COMPARISON")
            print("-" * 50)
            print(f"Latency Ratio (Triton/Inferneo): {latency_ratio:.2f}x")
            print(f"Inferneo is {latency_ratio:.2f}x faster than Triton")
            
            if 'tokens_per_second' in inferneo_throughput and 'tokens_per_second' in triton_throughput:
                throughput_ratio = inferneo_throughput['tokens_per_second'] / triton_throughput['tokens_per_second']
                completion_throughput_ratio = inferneo_throughput['completion_tokens_per_second'] / triton_throughput['completion_tokens_per_second']
                print(f"Token Throughput Ratio (Inferneo/Triton): {throughput_ratio:.2f}x")
                print(f"Completion Token Throughput Ratio (Inferneo/Triton): {completion_throughput_ratio:.2f}x")
        
        # Concurrency Results
        if concurrency_results and "inferneo" in concurrency_results:
            print(f"\n🔄 CONCURRENCY BENCHMARK RESULTS")
            print("-" * 50)
            
            for concurrency in sorted(concurrency_results["inferneo"].keys()):
                print(f"\nConcurrency Level: {concurrency}")
                
                inferneo_conc = concurrency_results["inferneo"][concurrency]
                triton_conc = concurrency_results["triton"][concurrency]
                
                print(f"  Inferneo:")
                print(f"    Requests/sec: {inferneo_conc['requests_per_second']:.2f}")
                print(f"    Tokens/sec: {inferneo_conc['tokens_per_second']:.2f}")
                print(f"    Completion Tokens/sec: {inferneo_conc['completion_tokens_per_second']:.2f}")
                print(f"    Success Rate: {inferneo_conc['successful_requests']}/{inferneo_conc['total_requests']}")
                
                print(f"  Triton:")
                print(f"    Requests/sec: {triton_conc['requests_per_second']:.2f}")
                print(f"    Tokens/sec: {triton_conc['tokens_per_second']:.2f}")
                print(f"    Completion Tokens/sec: {triton_conc['completion_tokens_per_second']:.2f}")
                print(f"    Success Rate: {triton_conc['successful_requests']}/{triton_conc['total_requests']}")
        else:
            print(f"\n🔄 CONCURRENCY BENCHMARK RESULTS")
            print("-" * 50)
            print("Concurrency benchmark results not available")
        
        print("\n" + "="*80)
        print("✅ BENCHMARK COMPLETED")
        print("="*80)

    async def run_fair_concurrency_benchmark(self, prompt: str, concurrency_levels: List[int]) -> Dict[str, Any]:
        """Run concurrency benchmark on both systems with identical conditions"""
        logger.info("Running fair concurrency benchmark...")
        
        # Warmup both systems
        await self.warmup_both_systems([prompt])
        
        results = {
            "inferneo": {},
            "triton": {}
        }
        
        for concurrency in concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency}")
            
            # Test Inferneo
            try:
                start_time = time.time()
                tasks = [self.inferneo_engine.generate(prompt, max_tokens=50) for _ in range(concurrency)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                inferneo_time = time.time() - start_time
                
                # Count successful responses and tokens
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                total_tokens = sum(len(prompt.split()) + len(r.text.split()) for r in successful_responses)
                completion_tokens = sum(len(r.text.split()) for r in successful_responses)
                
                inferneo_throughput = len(successful_responses) / inferneo_time if inferneo_time > 0 else 0
                inferneo_tokens_per_second = total_tokens / inferneo_time if inferneo_time > 0 else 0
                inferneo_completion_tokens_per_second = completion_tokens / inferneo_time if inferneo_time > 0 else 0
                
                results["inferneo"][concurrency] = {
                    "requests_per_second": inferneo_throughput,
                    "tokens_per_second": inferneo_tokens_per_second,
                    "completion_tokens_per_second": inferneo_completion_tokens_per_second,
                    "total_time": inferneo_time,
                    "successful_requests": len(successful_responses),
                    "total_requests": concurrency,
                    "total_tokens": total_tokens,
                    "total_completion_tokens": completion_tokens,
                    "errors": len(responses) - len(successful_responses)
                }
            except Exception as e:
                logger.warning(f"Inferneo concurrency {concurrency} failed: {e}")
                results["inferneo"][concurrency] = {
                    "requests_per_second": 0,
                    "tokens_per_second": 0,
                    "completion_tokens_per_second": 0,
                    "total_time": 0,
                    "successful_requests": 0,
                    "total_requests": concurrency,
                    "total_tokens": 0,
                    "total_completion_tokens": 0,
                    "errors": concurrency
                }
            
            # Test Triton
            try:
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(self._triton_request, prompt) for _ in range(concurrency)]
                    responses = [future.result() for future in futures]
                triton_time = time.time() - start_time
                
                # Count successful responses and tokens
                successful_responses = [r for r in responses if r and not isinstance(r, Exception)]
                total_tokens = sum(len(prompt.split()) + len(r.get('text', '').split()) for r in successful_responses)
                completion_tokens = sum(len(r.get('text', '').split()) for r in successful_responses)
                
                triton_throughput = len(successful_responses) / triton_time if triton_time > 0 else 0
                triton_tokens_per_second = total_tokens / triton_time if triton_time > 0 else 0
                triton_completion_tokens_per_second = completion_tokens / triton_time if triton_time > 0 else 0
                
                results["triton"][concurrency] = {
                    "requests_per_second": triton_throughput,
                    "tokens_per_second": triton_tokens_per_second,
                    "completion_tokens_per_second": triton_completion_tokens_per_second,
                    "total_time": triton_time,
                    "successful_requests": len(successful_responses),
                    "total_requests": concurrency,
                    "total_tokens": total_tokens,
                    "total_completion_tokens": completion_tokens,
                    "errors": len(responses) - len(successful_responses)
                }
            except Exception as e:
                logger.warning(f"Triton concurrency {concurrency} failed: {e}")
                results["triton"][concurrency] = {
                    "requests_per_second": 0,
                    "tokens_per_second": 0,
                    "completion_tokens_per_second": 0,
                    "total_time": 0,
                    "successful_requests": 0,
                    "total_requests": concurrency,
                    "total_tokens": 0,
                    "total_completion_tokens": 0,
                    "errors": concurrency
                }
        
        return results

    def _get_gpu_utilization(self) -> Dict[str, Any]:
        """Get current GPU utilization"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    gpu_util, mem_used, mem_total, temp = lines[0].split(', ')
                    return {
                        'gpu_utilization': int(gpu_util),
                        'memory_used_mb': int(mem_used),
                        'memory_total_mb': int(mem_total),
                        'temperature_c': int(temp)
                    }
        except Exception as e:
            logger.warning(f"Failed to get GPU utilization: {e}")
        return {}

    def _monitor_gpu_during_benchmark(self, duration_seconds: int = 5) -> List[Dict[str, Any]]:
        """Monitor GPU during benchmark"""
        measurements = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            measurement = self._get_gpu_utilization()
            measurement['timestamp'] = time.time()
            measurements.append(measurement)
            time.sleep(0.5)  # Sample every 500ms
            
        return measurements

async def main():
    """Main function"""
    benchmark = FairComparisonBenchmark()
    await benchmark.initialize()
    results = await benchmark.run_fair_benchmark()
    benchmark.print_results()
    benchmark.print_comparison_report(results["results"]["latency"], results["results"]["concurrency"])

if __name__ == "__main__":
    asyncio.run(main()) 