#!/usr/bin/env python3
"""
Enhanced Engine Demo for Inferneo

This script demonstrates the enhanced engine with dynamic batching,
showing how to use different batching strategies and monitor performance.
"""

import asyncio
import time
import random
import logging
from typing import List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inferneo.core.enhanced_engine import EnhancedInferneoEngine, EnhancedEngineConfig
from inferneo.core.scheduler import BatchingStrategy
from inferneo.models.base import ModelFormat
from inferneo.models.manager import ModelMetadata


class EnhancedEngineDemo:
    """Demo class for enhanced engine capabilities"""
    
    def __init__(self):
        # Configure enhanced engine
        config = EnhancedEngineConfig(
            model="gpt2",
            max_workers=4,
            max_memory_gb=8,
            enable_dynamic_batching=True,
            enable_speculative_decoding=True,
            enable_kv_cache_optimization=True,
            batching_strategy=BatchingStrategy.ADAPTIVE,
            enable_performance_monitoring=True,
            enable_health_checks=True
        )
        
        self.engine = EnhancedInferneoEngine(config)
        self.logger = logging.getLogger(__name__)
        
        # Demo data
        self.test_prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
            "To be or not to be, that is the question.",
            "All happy families are alike; each unhappy family is unhappy in its own way.",
            "Call me Ishmael.",
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
            "The only way to do great work is to love what you do.",
            "Life is what happens when you're busy making other plans.",
            "The future belongs to those who believe in the beauty of their dreams."
        ]
    
    async def run_demo(self):
        """Run the complete enhanced engine demo"""
        self.logger.info("🚀 Starting Enhanced Engine Demo")
        
        # Start engine
        await self.engine.start()
        
        # Register a dummy model for testing
        await self._setup_dummy_model()
        
        # Run different batching strategy demos
        await self.demo_adaptive_batching()
        await self.demo_coalescing_batching()
        await self.demo_latency_optimized_batching()
        await self.demo_throughput_optimized_batching()
        
        # Show final statistics
        await self.show_final_stats()
        
        # Stop engine
        await self.engine.stop()
        
        self.logger.info("✅ Enhanced Engine Demo completed")
    
    async def _setup_dummy_model(self):
        """Setup a dummy model for testing"""
        # Register a dummy model
        model_id = "gpt2:1"
        self.engine.model_manager.registered_models[model_id] = ModelMetadata(
            name="gpt2",
            version="1",
            path="models/huggingface/gpt2",
            format=ModelFormat.HUGGINGFACE,
            config={},
            metadata={},
            created_at=time.time()
        )
        
        # Add to models for active_models property
        self.engine.model_manager.models[model_id] = self.engine.model_manager.registered_models[model_id]
        
        # Load the model
        await self.engine.load_model("gpt2:1")
        self.engine.default_model = "gpt2:1"
        self.engine.loaded_models["gpt2:1"] = self.engine.model_manager.loaded_models["gpt2:1"]
    
    async def demo_adaptive_batching(self):
        """Demo adaptive batching strategy"""
        self.logger.info("\n📊 Demo 1: Adaptive Batching Strategy")
        self.logger.info("=" * 50)
        
        self.engine.set_batching_strategy(BatchingStrategy.ADAPTIVE)
        
        # Generate requests with varying characteristics
        prompts = self._generate_diverse_prompts(15)
        
        # Process requests
        start_time = time.time()
        results = await self.engine.generate_batch(prompts, max_tokens=50)
        total_time = time.time() - start_time
        
        # Show results
        self.logger.info(f"Adaptive Batching Results:")
        self.logger.info(f"  - Processed {len(results)} requests in {total_time:.3f}s")
        self.logger.info(f"  - Average time per request: {total_time/len(results):.3f}s")
        
        # Show batching stats
        batching_stats = self.engine.get_batching_stats()
        if batching_stats:
            self.logger.info(f"  - Average Batch Size: {batching_stats.get('avg_batch_size', 0):.2f}")
            self.logger.info(f"  - Average Throughput: {batching_stats.get('avg_throughput', 0):.2f} req/s")
            self.logger.info(f"  - Average Latency: {batching_stats.get('avg_latency', 0):.3f}s")
    
    async def demo_coalescing_batching(self):
        """Demo coalescing batching strategy"""
        self.logger.info("\n🔗 Demo 2: Coalescing Batching Strategy")
        self.logger.info("=" * 50)
        
        self.engine.set_batching_strategy(BatchingStrategy.COALESCING)
        
        # Generate requests with similar lengths for coalescing
        prompts = self._generate_similar_length_prompts(20)
        
        # Process requests
        start_time = time.time()
        results = await self.engine.generate_batch(prompts, max_tokens=40)
        total_time = time.time() - start_time
        
        # Show results
        self.logger.info(f"Coalescing Batching Results:")
        self.logger.info(f"  - Processed {len(results)} requests in {total_time:.3f}s")
        self.logger.info(f"  - Average time per request: {total_time/len(results):.3f}s")
        
        # Show batching stats
        batching_stats = self.engine.get_batching_stats()
        if batching_stats:
            self.logger.info(f"  - Average Batch Size: {batching_stats.get('avg_batch_size', 0):.2f}")
            self.logger.info(f"  - Coalescing Buckets: {batching_stats.get('coalescing_buckets', 0)}")
            self.logger.info(f"  - Average Throughput: {batching_stats.get('avg_throughput', 0):.2f} req/s")
    
    async def demo_latency_optimized_batching(self):
        """Demo latency-optimized batching strategy"""
        self.logger.info("\n⚡ Demo 3: Latency-Optimized Batching Strategy")
        self.logger.info("=" * 50)
        
        self.engine.set_batching_strategy(BatchingStrategy.LATENCY_OPTIMIZED)
        
        # Generate high-priority requests
        prompts = self._generate_high_priority_prompts(10)
        
        # Process requests with high priority
        start_time = time.time()
        results = await self.engine.generate_batch(prompts, max_tokens=30, priority=8)
        total_time = time.time() - start_time
        
        # Show results
        self.logger.info(f"Latency-Optimized Batching Results:")
        self.logger.info(f"  - Processed {len(results)} requests in {total_time:.3f}s")
        self.logger.info(f"  - Average time per request: {total_time/len(results):.3f}s")
        
        # Show batching stats
        batching_stats = self.engine.get_batching_stats()
        if batching_stats:
            self.logger.info(f"  - Average Batch Size: {batching_stats.get('avg_batch_size', 0):.2f}")
            self.logger.info(f"  - Average Latency: {batching_stats.get('avg_latency', 0):.3f}s")
            self.logger.info(f"  - Queue Size: {batching_stats.get('queue_size', 0)}")
    
    async def demo_throughput_optimized_batching(self):
        """Demo throughput-optimized batching strategy"""
        self.logger.info("\n🚀 Demo 4: Throughput-Optimized Batching Strategy")
        self.logger.info("=" * 50)
        
        self.engine.set_batching_strategy(BatchingStrategy.THROUGHPUT_OPTIMIZED)
        
        # Generate many requests for throughput testing
        prompts = self._generate_throughput_prompts(25)
        
        # Process requests
        start_time = time.time()
        results = await self.engine.generate_batch(prompts, max_tokens=60, priority=3)
        total_time = time.time() - start_time
        
        # Show results
        self.logger.info(f"Throughput-Optimized Batching Results:")
        self.logger.info(f"  - Processed {len(results)} requests in {total_time:.3f}s")
        self.logger.info(f"  - Average time per request: {total_time/len(results):.3f}s")
        
        # Show batching stats
        batching_stats = self.engine.get_batching_stats()
        if batching_stats:
            self.logger.info(f"  - Average Batch Size: {batching_stats.get('avg_batch_size', 0):.2f}")
            self.logger.info(f"  - Average Throughput: {batching_stats.get('avg_throughput', 0):.2f} req/s")
            self.logger.info(f"  - GPU Utilization: {batching_stats.get('avg_gpu_utilization', 0):.2f}")
    
    def _generate_diverse_prompts(self, count: int) -> List[str]:
        """Generate diverse prompts for adaptive batching"""
        prompts = []
        for i in range(count):
            base_prompt = random.choice(self.test_prompts)
            # Add some variation
            if random.random() < 0.3:
                base_prompt += " " + " ".join(["the", "and", "or", "but"][:random.randint(1, 3)])
            prompts.append(base_prompt)
        return prompts
    
    def _generate_similar_length_prompts(self, count: int) -> List[str]:
        """Generate prompts with similar lengths for coalescing"""
        prompts = []
        base_length = 50  # Target length
        
        for i in range(count):
            # Generate prompts with similar lengths
            prompt_length = base_length + random.randint(-5, 5)
            words = random.choice(self.test_prompts).split()
            
            # Adjust prompt to target length
            if len(words) > prompt_length:
                words = words[:prompt_length]
            else:
                # Pad with additional words
                while len(words) < prompt_length:
                    words.append(random.choice(["the", "and", "or", "but", "in", "on", "at"]))
            
            prompt = " ".join(words)
            prompts.append(prompt)
        
        return prompts
    
    def _generate_high_priority_prompts(self, count: int) -> List[str]:
        """Generate high-priority prompts for latency optimization"""
        prompts = []
        for i in range(count):
            base_prompt = random.choice(self.test_prompts)
            # Make them shorter for lower latency
            words = base_prompt.split()[:random.randint(5, 15)]
            prompt = " ".join(words)
            prompts.append(prompt)
        return prompts
    
    def _generate_throughput_prompts(self, count: int) -> List[str]:
        """Generate prompts optimized for throughput"""
        prompts = []
        for i in range(count):
            base_prompt = random.choice(self.test_prompts)
            # Medium length for throughput optimization
            words = base_prompt.split()[:random.randint(20, 40)]
            prompt = " ".join(words)
            prompts.append(prompt)
        return prompts
    
    async def show_final_stats(self):
        """Show final statistics"""
        self.logger.info("\n📈 Final Enhanced Engine Statistics")
        self.logger.info("=" * 50)
        
        stats = self.engine.get_enhanced_stats()
        
        self.logger.info(f"Engine Statistics:")
        self.logger.info(f"  - Models Loaded: {stats.get('models_loaded', 0)}")
        self.logger.info(f"  - Routing Rules: {stats.get('routing_rules', 0)}")
        self.logger.info(f"  - A/B Tests: {stats.get('ab_tests', 0)}")
        
        enhanced_features = stats.get('enhanced_features', {})
        self.logger.info(f"Enhanced Features:")
        self.logger.info(f"  - Dynamic Batching: {enhanced_features.get('dynamic_batching', False)}")
        self.logger.info(f"  - Speculative Decoding: {enhanced_features.get('speculative_decoding', False)}")
        self.logger.info(f"  - KV Cache Optimization: {enhanced_features.get('kv_cache_optimization', False)}")
        self.logger.info(f"  - Performance Monitoring: {enhanced_features.get('performance_monitoring', False)}")
        self.logger.info(f"  - Health Checks: {enhanced_features.get('health_checks', False)}")
        
        batching_stats = stats.get('batching', {})
        if batching_stats:
            self.logger.info(f"Batching Statistics:")
            self.logger.info(f"  - Current Batch Size: {batching_stats.get('current_batch_size', 0)}")
            self.logger.info(f"  - Batching Strategy: {batching_stats.get('batching_strategy', 'unknown')}")
            self.logger.info(f"  - Average Batch Size: {batching_stats.get('avg_batch_size', 0):.2f}")
            self.logger.info(f"  - Average Throughput: {batching_stats.get('avg_throughput', 0):.2f} req/s")
            self.logger.info(f"  - Average Latency: {batching_stats.get('avg_latency', 0):.3f}s")
            self.logger.info(f"  - GPU Utilization: {batching_stats.get('avg_gpu_utilization', 0):.2f}")


async def main():
    """Main demo function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run demo
    demo = EnhancedEngineDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main()) 