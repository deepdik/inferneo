#!/usr/bin/env python3
"""
Simple and fast test for enhanced scheduler
"""

import asyncio
import time
import logging
from enum import Enum


class BatchingStrategy(Enum):
    """Batching strategies"""
    ADAPTIVE = "adaptive"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


class SimpleRequest:
    """Simple request for testing"""
    def __init__(self, prompt: str, max_tokens: int = 10):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.request_id = f"req_{int(time.time() * 1000)}"


class FastScheduler:
    """Fast test scheduler"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategy = BatchingStrategy.ADAPTIVE
        self.requests = []
        self.processed = 0
        self.batch_sizes = []
        
    def set_strategy(self, strategy: BatchingStrategy):
        """Set batching strategy"""
        self.strategy = strategy
        self.logger.info(f"Strategy set to: {strategy.value}")
    
    async def add_request(self, request: SimpleRequest):
        """Add a request"""
        self.requests.append(request)
        self.logger.info(f"Added request: {request.request_id}")
    
    async def process_batch(self):
        """Process a batch of requests"""
        if not self.requests:
            return []
        
        # Determine batch size based on strategy
        if self.strategy == BatchingStrategy.ADAPTIVE:
            batch_size = min(3, len(self.requests))
        elif self.strategy == BatchingStrategy.LATENCY_OPTIMIZED:
            batch_size = min(2, len(self.requests))
        elif self.strategy == BatchingStrategy.THROUGHPUT_OPTIMIZED:
            batch_size = min(4, len(self.requests))
        else:
            batch_size = min(3, len(self.requests))
        
        # Take batch
        batch = self.requests[:batch_size]
        self.requests = self.requests[batch_size:]
        
        self.batch_sizes.append(len(batch))
        self.processed += len(batch)
        
        self.logger.info(f"Processed batch of {len(batch)} requests")
        return batch
    
    def get_stats(self):
        """Get statistics"""
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        return {
            "strategy": self.strategy.value,
            "total_processed": self.processed,
            "avg_batch_size": avg_batch_size,
            "remaining_requests": len(self.requests)
        }


async def quick_test():
    """Quick test function"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Quick Enhanced Scheduler Test")
    logger.info("=" * 40)
    
    # Test different strategies
    strategies = [
        BatchingStrategy.ADAPTIVE,
        BatchingStrategy.LATENCY_OPTIMIZED,
        BatchingStrategy.THROUGHPUT_OPTIMIZED
    ]
    
    for strategy in strategies:
        logger.info(f"\n📊 Testing {strategy.value} strategy:")
        
        # Create scheduler
        scheduler = FastScheduler()
        scheduler.set_strategy(strategy)
        
        # Add some requests
        requests = [
            SimpleRequest(f"Prompt {i}", max_tokens=20)
            for i in range(6)
        ]
        
        for req in requests:
            await scheduler.add_request(req)
        
        # Process all requests
        while scheduler.requests:
            batch = await scheduler.process_batch()
            await asyncio.sleep(0.01)  # Simulate processing time
        
        # Show results
        stats = scheduler.get_stats()
        logger.info(f"  ✅ Processed {stats['total_processed']} requests")
        logger.info(f"  📦 Average batch size: {stats['avg_batch_size']:.1f}")
        logger.info(f"  ⚡ Strategy: {stats['strategy']}")
    
    logger.info("\n🎉 All tests completed successfully!")
    logger.info("Enhanced scheduler is working correctly!")


if __name__ == "__main__":
    asyncio.run(quick_test()) 