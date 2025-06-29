#!/usr/bin/env python3
"""
Dynamic Batching Demo for Inferneo

This script demonstrates the enhanced dynamic batching capabilities
including adaptive sizing, request coalescing, and performance monitoring.
"""

import asyncio
import time
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inferneo.core.enhanced_scheduler import (
    EnhancedScheduler, BatchingStrategy, ScheduledRequest
)
from inferneo.core.config import SchedulerConfig


@dataclass
class DemoRequest:
    """Demo request for testing"""
    request_id: str
    prompt: str
    max_tokens: int
    priority: int = 5


class DynamicBatchingDemo:
    """Demo class for dynamic batching capabilities"""
    
    def __init__(self):
        # Configure scheduler
        config = SchedulerConfig(
            max_num_partial_prefills=4,
            max_waiting_tokens=2048,
            enable_priority_queue=True,
            preemption_mode="recompute"
        )
        
        self.scheduler = EnhancedScheduler(config)
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
        """Run the complete dynamic batching demo"""
        self.logger.info("🚀 Starting Dynamic Batching Demo")
        
        # Initialize scheduler
        await self.scheduler.initialize()
        
        # Run different batching strategy demos
        await self.demo_adaptive_batching()
        await self.demo_coalescing_batching()
        await self.demo_latency_optimized_batching()
        await self.demo_throughput_optimized_batching()
        
        # Show final statistics
        await self.show_final_stats()
        
        # Cleanup
        await self.scheduler.cleanup()
        
        self.logger.info("✅ Dynamic Batching Demo completed")
    
    async def demo_adaptive_batching(self):
        """Demo adaptive batching strategy"""
        self.logger.info("\n📊 Demo 1: Adaptive Batching Strategy")
        self.logger.info("=" * 50)
        
        self.scheduler.set_batching_strategy(BatchingStrategy.ADAPTIVE)
        
        # Generate requests with varying characteristics
        requests = self._generate_diverse_requests(20)
        
        # Add requests to scheduler
        for request in requests:
            await self.scheduler.add_request(request)
            await asyncio.sleep(0.1)  # Simulate request arrival
        
        # Process batches
        await self._process_batches("Adaptive")
        
        # Show stats
        stats = self.scheduler.get_batching_stats()
        self.logger.info(f"Adaptive Batching Results:")
        self.logger.info(f"  - Average Batch Size: {stats.get('avg_batch_size', 0):.2f}")
        self.logger.info(f"  - Average Throughput: {stats.get('avg_throughput', 0):.2f} req/s")
        self.logger.info(f"  - Average Latency: {stats.get('avg_latency', 0):.3f}s")
    
    async def demo_coalescing_batching(self):
        """Demo coalescing batching strategy"""
        self.logger.info("\n🔗 Demo 2: Coalescing Batching Strategy")
        self.logger.info("=" * 50)
        
        self.scheduler.set_batching_strategy(BatchingStrategy.COALESCING)
        
        # Generate requests with similar lengths for coalescing
        requests = self._generate_similar_length_requests(25)
        
        # Add requests to scheduler
        for request in requests:
            await self.scheduler.add_request(request)
            await asyncio.sleep(0.05)  # Faster arrival for coalescing
        
        # Process batches
        await self._process_batches("Coalescing")
        
        # Show stats
        stats = self.scheduler.get_batching_stats()
        self.logger.info(f"Coalescing Batching Results:")
        self.logger.info(f"  - Average Batch Size: {stats.get('avg_batch_size', 0):.2f}")
        self.logger.info(f"  - Coalescing Buckets: {stats.get('coalescing_buckets', 0)}")
        self.logger.info(f"  - Average Throughput: {stats.get('avg_throughput', 0):.2f} req/s")
    
    async def demo_latency_optimized_batching(self):
        """Demo latency-optimized batching strategy"""
        self.logger.info("\n⚡ Demo 3: Latency-Optimized Batching Strategy")
        self.logger.info("=" * 50)
        
        self.scheduler.set_batching_strategy(BatchingStrategy.LATENCY_OPTIMIZED)
        
        # Generate high-priority requests
        requests = self._generate_high_priority_requests(15)
        
        # Add requests to scheduler
        for request in requests:
            await self.scheduler.add_request(request)
            await asyncio.sleep(0.2)  # Slower arrival for latency focus
        
        # Process batches
        await self._process_batches("Latency-Optimized")
        
        # Show stats
        stats = self.scheduler.get_batching_stats()
        self.logger.info(f"Latency-Optimized Batching Results:")
        self.logger.info(f"  - Average Batch Size: {stats.get('avg_batch_size', 0):.2f}")
        self.logger.info(f"  - Average Latency: {stats.get('avg_latency', 0):.3f}s")
        self.logger.info(f"  - Queue Size: {stats.get('queue_size', 0)}")
    
    async def demo_throughput_optimized_batching(self):
        """Demo throughput-optimized batching strategy"""
        self.logger.info("\n🚀 Demo 4: Throughput-Optimized Batching Strategy")
        self.logger.info("=" * 50)
        
        self.scheduler.set_batching_strategy(BatchingStrategy.THROUGHPUT_OPTIMIZED)
        
        # Generate many requests for throughput testing
        requests = self._generate_throughput_requests(30)
        
        # Add requests to scheduler rapidly
        for request in requests:
            await self.scheduler.add_request(request)
            await asyncio.sleep(0.02)  # Very fast arrival for throughput
        
        # Process batches
        await self._process_batches("Throughput-Optimized")
        
        # Show stats
        stats = self.scheduler.get_batching_stats()
        self.logger.info(f"Throughput-Optimized Batching Results:")
        self.logger.info(f"  - Average Batch Size: {stats.get('avg_batch_size', 0):.2f}")
        self.logger.info(f"  - Average Throughput: {stats.get('avg_throughput', 0):.2f} req/s")
        self.logger.info(f"  - GPU Utilization: {stats.get('avg_gpu_utilization', 0):.2f}")
    
    async def _process_batches(self, strategy_name: str):
        """Process batches from scheduler"""
        batch_count = 0
        total_requests = 0
        
        while True:
            # Get next batch
            batch = await self.scheduler.get_next_batch()
            
            if not batch:
                # No more batches, wait a bit and check again
                await asyncio.sleep(0.1)
                if len(self.scheduler._waiting_queue) == 0:
                    break
                continue
            
            batch_count += 1
            total_requests += len(batch)
            
            # Simulate batch processing
            processing_time = self._simulate_batch_processing(batch)
            gpu_utilization = random.uniform(0.6, 0.95)
            memory_usage = random.uniform(0.4, 0.8)
            
            # Complete batch
            batch_id = batch[0].batch_id if batch else f"batch_{batch_count}"
            await self.scheduler.complete_batch(batch_id, processing_time, gpu_utilization, memory_usage)
            
            # Complete individual requests
            for request in batch:
                await self.scheduler.complete_request(
                    request.request_id, 
                    success=True,
                    processing_time=processing_time / len(batch),
                    gpu_utilization=gpu_utilization
                )
            
            self.logger.info(f"{strategy_name} Batch {batch_count}: "
                           f"{len(batch)} requests, {processing_time:.3f}s, "
                           f"GPU: {gpu_utilization:.2f}")
            
            # Small delay between batches
            await asyncio.sleep(0.05)
        
        self.logger.info(f"{strategy_name}: Processed {batch_count} batches, {total_requests} total requests")
    
    def _simulate_batch_processing(self, batch: List[ScheduledRequest]) -> float:
        """Simulate batch processing time"""
        # Base processing time
        base_time = 0.1
        
        # Add time based on batch size
        batch_size_factor = len(batch) * 0.02
        
        # Add time based on total tokens
        total_tokens = sum(req.estimated_tokens for req in batch)
        token_factor = total_tokens * 0.001
        
        # Add some randomness
        random_factor = random.uniform(0.8, 1.2)
        
        return (base_time + batch_size_factor + token_factor) * random_factor
    
    def _generate_diverse_requests(self, count: int) -> List[DemoRequest]:
        """Generate diverse requests for adaptive batching"""
        requests = []
        for i in range(count):
            prompt = random.choice(self.test_prompts)
            max_tokens = random.randint(10, 100)
            priority = random.randint(1, 10)
            
            request = DemoRequest(
                request_id=f"req_{i:03d}",
                prompt=prompt,
                max_tokens=max_tokens,
                priority=priority
            )
            requests.append(request)
        
        return requests
    
    def _generate_similar_length_requests(self, count: int) -> List[DemoRequest]:
        """Generate requests with similar lengths for coalescing"""
        requests = []
        base_length = 50  # Target length
        
        for i in range(count):
            # Generate prompts with similar lengths
            prompt_length = base_length + random.randint(-10, 10)
            words = random.choice(self.test_prompts).split()
            
            # Adjust prompt to target length
            if len(words) > prompt_length:
                words = words[:prompt_length]
            else:
                # Pad with additional words
                while len(words) < prompt_length:
                    words.append(random.choice(["the", "and", "or", "but", "in", "on", "at"]))
            
            prompt = " ".join(words)
            max_tokens = random.randint(20, 60)
            
            request = DemoRequest(
                request_id=f"coalesce_{i:03d}",
                prompt=prompt,
                max_tokens=max_tokens,
                priority=5
            )
            requests.append(request)
        
        return requests
    
    def _generate_high_priority_requests(self, count: int) -> List[DemoRequest]:
        """Generate high-priority requests for latency optimization"""
        requests = []
        for i in range(count):
            prompt = random.choice(self.test_prompts)
            max_tokens = random.randint(5, 30)  # Shorter for lower latency
            priority = random.randint(8, 10)  # High priority
            
            request = DemoRequest(
                request_id=f"latency_{i:03d}",
                prompt=prompt,
                max_tokens=max_tokens,
                priority=priority
            )
            requests.append(request)
        
        return requests
    
    def _generate_throughput_requests(self, count: int) -> List[DemoRequest]:
        """Generate requests optimized for throughput"""
        requests = []
        for i in range(count):
            prompt = random.choice(self.test_prompts)
            max_tokens = random.randint(30, 80)  # Medium length
            priority = random.randint(1, 7)  # Lower priority for batching
            
            request = DemoRequest(
                request_id=f"throughput_{i:03d}",
                prompt=prompt,
                max_tokens=max_tokens,
                priority=priority
            )
            requests.append(request)
        
        return requests
    
    async def show_final_stats(self):
        """Show final statistics"""
        self.logger.info("\n📈 Final Statistics")
        self.logger.info("=" * 50)
        
        stats = self.scheduler.get_stats()
        
        self.logger.info(f"Total Requests Processed: {stats['total_requests']}")
        self.logger.info(f"Completed Requests: {stats['completed_requests']}")
        self.logger.info(f"Average Wait Time: {stats['avg_wait_time']:.3f}s")
        self.logger.info(f"Average Processing Time: {stats['avg_processing_time']:.3f}s")
        
        batching_stats = self.scheduler.get_batching_stats()
        if batching_stats:
            self.logger.info(f"Average Batch Size: {batching_stats['avg_batch_size']:.2f}")
            self.logger.info(f"Average Throughput: {batching_stats['avg_throughput']:.2f} req/s")
            self.logger.info(f"Average Latency: {batching_stats['avg_latency']:.3f}s")
            self.logger.info(f"Average GPU Utilization: {batching_stats['avg_gpu_utilization']:.2f}")


async def main():
    """Main demo function"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run demo
    demo = DynamicBatchingDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
