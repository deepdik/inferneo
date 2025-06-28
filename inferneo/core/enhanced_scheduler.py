"""
Enhanced Dynamic Batching Scheduler for Inferneo

Advanced request scheduler with intelligent batching, adaptive sizing,
request coalescing, and GPU utilization optimization.
"""

import asyncio
import heapq
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import logging
import numpy as np

from .config import SchedulerConfig


class RequestState(Enum):
    """Request states"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchingStrategy(Enum):
    """Batching strategies"""
    FIXED_SIZE = "fixed_size"
    ADAPTIVE = "adaptive"
    COALESCING = "coalescing"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


@dataclass
class BatchMetrics:
    """Metrics for batch performance"""
    batch_size: int
    total_tokens: int
    processing_time: float
    gpu_utilization: float
    memory_usage: float
    throughput: float
    latency: float
    created_at: float


@dataclass
class ScheduledRequest:
    """Request with enhanced scheduling metadata"""
    request_id: str
    priority: int
    created_at: float
    estimated_tokens: int
    prompt_length: int
    max_tokens: int
    state: RequestState = RequestState.WAITING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    batch_id: Optional[str] = None
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class GPUMonitor:
    """Simple GPU utilization monitor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._last_check = 0
        self._cached_utilization = 0.5
    
    async def get_utilization(self) -> float:
        """Get current GPU utilization (simplified implementation)"""
        current_time = time.time()
        
        # Cache for 1 second
        if current_time - self._last_check > 1.0:
            try:
                # Try to get real GPU utilization
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._cached_utilization = util.gpu / 100.0
            except ImportError:
                # Fallback to CPU-based estimation
                self._cached_utilization = self._estimate_gpu_utilization()
            except Exception as e:
                self.logger.debug(f"GPU monitoring error: {e}")
                # Keep previous value
            
            self._last_check = current_time
        
        return self._cached_utilization
    
    def _estimate_gpu_utilization(self) -> float:
        """Estimate GPU utilization based on system load"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Rough estimation: higher CPU usage might indicate GPU usage too
            return min(cpu_percent / 100.0 * 0.8 + 0.2, 0.95)
        except:
            return 0.5


class EnhancedScheduler:
    """
    Enhanced dynamic batching scheduler with adaptive sizing and intelligent coalescing
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Batching strategy
        self.batching_strategy = BatchingStrategy.ADAPTIVE
        self.adaptive_batch_size = config.max_num_partial_prefills
        self.min_batch_size = 1
        self.max_batch_size = config.max_num_partial_prefills * 2
        
        # Request queues
        self._waiting_queue: List[ScheduledRequest] = []
        self._running_requests: Dict[str, ScheduledRequest] = {}
        self._completed_requests: Dict[str, ScheduledRequest] = {}
        
        # Batching state
        self._current_batch: List[ScheduledRequest] = []
        self._batch_tokens = 0
        self._max_batch_tokens = config.max_waiting_tokens
        self._batch_id_counter = 0
        
        # Performance tracking
        self._total_requests = 0
        self._total_wait_time = 0.0
        self._total_processing_time = 0.0
        self._batch_metrics: List[BatchMetrics] = []
        self._recent_batch_metrics: List[BatchMetrics] = []
        
        # GPU monitoring
        self._gpu_monitor = GPUMonitor()
        self._last_gpu_check = 0
        self._gpu_utilization_history: List[float] = []
        
        # Coalescing buckets
        self._coalescing_buckets: Dict[int, List[ScheduledRequest]] = {}
        self._bucket_size = 64  # Token bucket size for coalescing
        
        # Async state
        self._is_running = False
        self._batch_ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        
        # Background tasks
        self._metrics_collection_task: Optional[asyncio.Task] = None
        self._adaptive_sizing_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the scheduler"""
        self._is_running = True
        
        # Start background tasks
        self._metrics_collection_task = asyncio.create_task(self._collect_metrics_loop())
        self._adaptive_sizing_task = asyncio.create_task(self._adaptive_sizing_loop())
        
        self.logger.info("Enhanced Dynamic Batching Scheduler initialized")
    
    async def add_request(self, request: Any) -> str:
        """
        Add a request to the scheduler with enhanced metadata
        
        Args:
            request: GenerationRequest object
            
        Returns:
            Request ID
        """
        # Estimate tokens for this request
        estimated_tokens = self._estimate_tokens(request)
        prompt_length = len(request.prompt.split())
        
        # Create scheduled request with enhanced metadata
        scheduled_request = ScheduledRequest(
            request_id=request.request_id,
            priority=getattr(request, 'priority', 5),
            created_at=time.time(),
            estimated_tokens=estimated_tokens,
            prompt_length=prompt_length,
            max_tokens=request.max_tokens
        )
        
        # Add to waiting queue
        heapq.heappush(self._waiting_queue, scheduled_request)
        self._total_requests += 1
        
        # Add to coalescing bucket
        self._add_to_coalescing_bucket(scheduled_request)
        
        # Try to form a batch
        await self._try_form_batch()
        
        return request.request_id
    
    async def get_next_batch(self) -> List[Any]:
        """
        Get the next batch of requests to process with dynamic sizing
        
        Returns:
            List of requests to process
        """
        if not self._current_batch:
            # Wait for batch to be ready or timeout
            try:
                await asyncio.wait_for(
                    self._batch_ready_event.wait(),
                    timeout=self._get_dynamic_timeout()
                )
            except asyncio.TimeoutError:
                return []
        
        # Return current batch and clear it
        batch = self._current_batch.copy()
        batch_id = f"batch_{self._batch_id_counter}"
        self._batch_id_counter += 1
        
        # Assign batch ID to requests
        for req in batch:
            req.batch_id = batch_id
        
        self._current_batch.clear()
        self._batch_tokens = 0
        self._batch_ready_event.clear()
        
        # Move requests to running state
        for scheduled_request in batch:
            scheduled_request.state = RequestState.RUNNING
            scheduled_request.start_time = time.time()
            self._running_requests[scheduled_request.request_id] = scheduled_request
        
        return batch
    
    async def complete_request(self, request_id: str, success: bool = True, 
                             processing_time: float = None, gpu_utilization: float = None):
        """
        Mark a request as completed with performance metrics
        
        Args:
            request_id: Request ID
            success: Whether the request completed successfully
            processing_time: Actual processing time
            gpu_utilization: GPU utilization during processing
        """
        if request_id not in self._running_requests:
            return
        
        scheduled_request = self._running_requests.pop(request_id)
        scheduled_request.end_time = time.time()
        
        if success:
            scheduled_request.state = RequestState.COMPLETED
            self._completed_requests[request_id] = scheduled_request
            
            # Update processing time if provided
            if processing_time is not None:
                self._total_processing_time += processing_time
            else:
                self._total_processing_time += scheduled_request.end_time - scheduled_request.start_time
            
            # Update GPU utilization history
            if gpu_utilization is not None:
                self._gpu_utilization_history.append(gpu_utilization)
                if len(self._gpu_utilization_history) > 100:
                    self._gpu_utilization_history.pop(0)
        
        else:
            scheduled_request.state = RequestState.FAILED
    
    async def complete_batch(self, batch_id: str, processing_time: float, 
                           gpu_utilization: float, memory_usage: float):
        """
        Record batch completion metrics for adaptive sizing
        
        Args:
            batch_id: Batch identifier
            processing_time: Total processing time for the batch
            gpu_utilization: Average GPU utilization during processing
            memory_usage: Memory usage during processing
        """
        # Find requests in this batch
        batch_requests = [req for req in self._running_requests.values() 
                         if req.batch_id == batch_id]
        
        if not batch_requests:
            return
        
        batch_size = len(batch_requests)
        total_tokens = sum(req.estimated_tokens for req in batch_requests)
        
        # Calculate metrics
        throughput = batch_size / processing_time if processing_time > 0 else 0
        avg_latency = processing_time / batch_size if batch_size > 0 else 0
        
        # Create batch metrics
        batch_metrics = BatchMetrics(
            batch_size=batch_size,
            total_tokens=total_tokens,
            processing_time=processing_time,
            gpu_utilization=gpu_utilization,
            memory_usage=memory_usage,
            throughput=throughput,
            latency=avg_latency,
            created_at=time.time()
        )
        
        # Store metrics
        self._batch_metrics.append(batch_metrics)
        self._recent_batch_metrics.append(batch_metrics)
        
        # Keep only recent metrics for adaptive sizing
        if len(self._recent_batch_metrics) > 50:
            self._recent_batch_metrics.pop(0)
        
        # Keep only last 1000 metrics for historical analysis
        if len(self._batch_metrics) > 1000:
            self._batch_metrics.pop(0)
    
    def _add_to_coalescing_bucket(self, request: ScheduledRequest):
        """Add request to appropriate coalescing bucket"""
        bucket_key = (request.prompt_length // self._bucket_size) * self._bucket_size
        if bucket_key not in self._coalescing_buckets:
            self._coalescing_buckets[bucket_key] = []
        self._coalescing_buckets[bucket_key].append(request)
    
    async def _try_form_batch(self):
        """Try to form a batch using dynamic batching strategy"""
        if not self._waiting_queue:
            return
        
        if self.batching_strategy == BatchingStrategy.COALESCING:
            await self._form_coalesced_batch()
        elif self.batching_strategy == BatchingStrategy.ADAPTIVE:
            await self._form_adaptive_batch()
        elif self.batching_strategy == BatchingStrategy.LATENCY_OPTIMIZED:
            await self._form_latency_optimized_batch()
        elif self.batching_strategy == BatchingStrategy.THROUGHPUT_OPTIMIZED:
            await self._form_throughput_optimized_batch()
        else:
            await self._form_fixed_size_batch()
        
        # Signal that batch is ready
        if self._current_batch:
            self._batch_ready_event.set()
    
    async def _form_adaptive_batch(self):
        """Form batch using adaptive sizing based on performance metrics"""
        # Get current GPU utilization
        gpu_util = await self._get_gpu_utilization()
        
        # Adjust batch size based on GPU utilization
        if gpu_util < 0.6:  # Under-utilized
            target_batch_size = min(self.adaptive_batch_size + 2, self.max_batch_size)
        elif gpu_util > 0.9:  # Over-utilized
            target_batch_size = max(self.adaptive_batch_size - 1, self.min_batch_size)
        else:
            target_batch_size = self.adaptive_batch_size
        
        # Form batch with target size
        while (self._waiting_queue and 
               len(self._current_batch) < target_batch_size and
               self._batch_tokens < self._max_batch_tokens):
            
            scheduled_request = heapq.heappop(self._waiting_queue)
            
            # Check if adding this request would exceed limits
            if (self._batch_tokens + scheduled_request.estimated_tokens > 
                self._max_batch_tokens):
                heapq.heappush(self._waiting_queue, scheduled_request)
                break
            
            # Add to current batch
            self._current_batch.append(scheduled_request)
            self._batch_tokens += scheduled_request.estimated_tokens
    
    async def _form_coalesced_batch(self):
        """Form batch by coalescing similar-length requests"""
        # Find the bucket with the most requests
        best_bucket = None
        max_requests = 0
        
        for bucket_key, requests in self._coalescing_buckets.items():
            if len(requests) > max_requests:
                max_requests = len(requests)
                best_bucket = bucket_key
        
        if not best_bucket or max_requests == 0:
            # Fall back to adaptive batching
            await self._form_adaptive_batch()
            return
        
        # Take requests from the best bucket
        bucket_requests = self._coalescing_buckets[best_bucket]
        requests_to_batch = min(len(bucket_requests), self.adaptive_batch_size)
        
        for _ in range(requests_to_batch):
            if not bucket_requests:
                break
            
            request = bucket_requests.pop(0)
            
            # Remove from waiting queue
            for i, waiting_req in enumerate(self._waiting_queue):
                if waiting_req.request_id == request.request_id:
                    self._waiting_queue.pop(i)
                    heapq.heapify(self._waiting_queue)
                    break
            
            # Add to current batch
            if (self._batch_tokens + request.estimated_tokens <= 
                self._max_batch_tokens):
                self._current_batch.append(request)
                self._batch_tokens += request.estimated_tokens
            else:
                # Put back in waiting queue
                heapq.heappush(self._waiting_queue, request)
                break
        
        # Clean up empty buckets
        if not bucket_requests:
            del self._coalescing_buckets[best_bucket]
    
    async def _form_latency_optimized_batch(self):
        """Form batch optimized for low latency"""
        # Prioritize small batches for low latency
        target_batch_size = max(1, self.adaptive_batch_size // 2)
        
        while (self._waiting_queue and 
               len(self._current_batch) < target_batch_size and
               self._batch_tokens < self._max_batch_tokens // 2):  # More conservative token limit
            
            scheduled_request = heapq.heappop(self._waiting_queue)
            
            if (self._batch_tokens + scheduled_request.estimated_tokens > 
                self._max_batch_tokens // 2):
                heapq.heappush(self._waiting_queue, scheduled_request)
                break
            
            self._current_batch.append(scheduled_request)
            self._batch_tokens += scheduled_request.estimated_tokens
    
    async def _form_throughput_optimized_batch(self):
        """Form batch optimized for high throughput"""
        # Use larger batches for high throughput
        target_batch_size = min(self.adaptive_batch_size * 2, self.max_batch_size)
        
        while (self._waiting_queue and 
               len(self._current_batch) < target_batch_size and
               self._batch_tokens < self._max_batch_tokens):
            
            scheduled_request = heapq.heappop(self._waiting_queue)
            
            if (self._batch_tokens + scheduled_request.estimated_tokens > 
                self._max_batch_tokens):
                heapq.heappush(self._waiting_queue, scheduled_request)
                break
            
            self._current_batch.append(scheduled_request)
            self._batch_tokens += scheduled_request.estimated_tokens
    
    async def _form_fixed_size_batch(self):
        """Form batch with fixed size (original behavior)"""
        while (self._waiting_queue and 
               len(self._current_batch) < self.config.max_num_partial_prefills and
               self._batch_tokens < self._max_batch_tokens):
            
            scheduled_request = heapq.heappop(self._waiting_queue)
            
            if (self._batch_tokens + scheduled_request.estimated_tokens > 
                self._max_batch_tokens):
                heapq.heappush(self._waiting_queue, scheduled_request)
                break
            
            self._current_batch.append(scheduled_request)
            self._batch_tokens += scheduled_request.estimated_tokens
    
    def _get_dynamic_timeout(self) -> float:
        """Get dynamic timeout based on queue state and performance"""
        queue_size = len(self._waiting_queue)
        
        if queue_size == 0:
            return 0.1  # Short timeout when no requests
        
        if queue_size > 10:
            return 0.01  # Very short timeout when queue is full
        
        if queue_size > 5:
            return 0.05  # Short timeout when queue is moderate
        
        return 0.1  # Default timeout
    
    async def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization"""
        current_time = time.time()
        
        # Cache GPU utilization for 1 second
        if current_time - self._last_gpu_check > 1.0:
            self._last_gpu_check = current_time
            gpu_util = await self._gpu_monitor.get_utilization()
            self._gpu_utilization_history.append(gpu_util)
            
            # Keep only recent history
            if len(self._gpu_utilization_history) > 10:
                self._gpu_utilization_history.pop(0)
        
        # Return average of recent measurements
        if self._gpu_utilization_history:
            return np.mean(self._gpu_utilization_history)
        
        return 0.5  # Default to 50% if no measurements
    
    def _estimate_tokens(self, request: Any) -> int:
        """
        Estimate the number of tokens for a request
        
        Args:
            request: GenerationRequest object
            
        Returns:
            Estimated token count
        """
        # Simple estimation based on prompt length and max tokens
        prompt_tokens = len(request.prompt.split())  # Rough estimate
        total_tokens = prompt_tokens + request.max_tokens
        
        # Apply some safety margin
        return min(total_tokens * 1.2, self.config.max_waiting_tokens)
    
    async def _collect_metrics_loop(self):
        """Background task to collect performance metrics"""
        while self._is_running:
            try:
                # Collect system metrics
                gpu_util = await self._get_gpu_utilization()
                
                # Log metrics periodically
                if len(self._recent_batch_metrics) % 10 == 0:
                    self.logger.debug(f"GPU Utilization: {gpu_util:.2f}, "
                                    f"Queue Size: {len(self._waiting_queue)}, "
                                    f"Batch Size: {self.adaptive_batch_size}")
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    async def _adaptive_sizing_loop(self):
        """Background task to adjust batch size based on performance"""
        while self._is_running:
            try:
                if len(self._recent_batch_metrics) >= 10:
                    await self._adjust_batch_size()
                
                await asyncio.sleep(30)  # Adjust every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in adaptive sizing: {e}")
                await asyncio.sleep(30)
    
    async def _adjust_batch_size(self):
        """Adjust batch size based on recent performance metrics"""
        if not self._recent_batch_metrics:
            return
        
        # Calculate performance metrics
        recent_metrics = self._recent_batch_metrics[-10:]  # Last 10 batches
        
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_latency = np.mean([m.latency for m in recent_metrics])
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])
        
        # Get current GPU utilization
        current_gpu_util = await self._get_gpu_utilization()
        
        # Adjust batch size based on performance
        old_batch_size = self.adaptive_batch_size
        
        if avg_gpu_util < 0.7 and current_gpu_util < 0.8:
            # Under-utilized, increase batch size
            self.adaptive_batch_size = min(self.adaptive_batch_size + 1, self.max_batch_size)
        elif avg_gpu_util > 0.9 or current_gpu_util > 0.95:
            # Over-utilized, decrease batch size
            self.adaptive_batch_size = max(self.adaptive_batch_size - 1, self.min_batch_size)
        elif avg_latency > 1.0:  # High latency
            # Reduce batch size for lower latency
            self.adaptive_batch_size = max(self.adaptive_batch_size - 1, self.min_batch_size)
        
        if old_batch_size != self.adaptive_batch_size:
            self.logger.info(f"Adjusted batch size: {old_batch_size} -> {self.adaptive_batch_size} "
                           f"(GPU: {current_gpu_util:.2f}, Latency: {avg_latency:.3f}s)")
    
    def set_batching_strategy(self, strategy: BatchingStrategy):
        """Set the batching strategy"""
        self.batching_strategy = strategy
        self.logger.info(f"Batching strategy changed to: {strategy.value}")
    
    def get_batching_stats(self) -> Dict[str, Any]:
        """Get detailed batching statistics"""
        if not self._recent_batch_metrics:
            return {}
        
        recent_metrics = self._recent_batch_metrics[-20:]  # Last 20 batches
        
        return {
            "current_batch_size": self.adaptive_batch_size,
            "batching_strategy": self.batching_strategy.value,
            "avg_batch_size": np.mean([m.batch_size for m in recent_metrics]),
            "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "avg_latency": np.mean([m.latency for m in recent_metrics]),
            "avg_gpu_utilization": np.mean([m.gpu_utilization for m in recent_metrics]),
            "coalescing_buckets": len(self._coalescing_buckets),
            "queue_size": len(self._waiting_queue),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        current_time = time.time()
        
        # Calculate average wait times
        avg_wait_time = 0
        avg_processing_time = 0
        
        if self._completed_requests:
            total_wait_time = sum(
                req.start_time - req.created_at 
                for req in self._completed_requests.values() 
                if req.start_time
            )
            avg_wait_time = total_wait_time / len(self._completed_requests)
            
            total_processing_time = sum(
                req.end_time - req.start_time 
                for req in self._completed_requests.values() 
                if req.start_time and req.end_time
            )
            avg_processing_time = total_processing_time / len(self._completed_requests)
        
        return {
            "total_requests": self._total_requests,
            "waiting_requests": len(self._waiting_queue),
            "running_requests": len(self._running_requests),
            "completed_requests": len(self._completed_requests),
            "current_batch_size": len(self._current_batch),
            "current_batch_tokens": self._batch_tokens,
            "avg_wait_time": avg_wait_time,
            "avg_processing_time": avg_processing_time,
            "queue_depth": len(self._waiting_queue),
            **self.get_batching_stats(),
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a request"""
        # Check running requests
        if request_id in self._running_requests:
            scheduled_request = self._running_requests.pop(request_id)
            scheduled_request.state = RequestState.CANCELLED
            return True
        
        # Check waiting queue
        for i, scheduled_request in enumerate(self._waiting_queue):
            if scheduled_request.request_id == request_id:
                self._waiting_queue.pop(i)
                heapq.heapify(self._waiting_queue)
                scheduled_request.state = RequestState.CANCELLED
                return True
        
        return False
    
    async def stop(self):
        """Stop the scheduler"""
        self._is_running = False
        self._stop_event.set()
        
        # Cancel background tasks
        if self._metrics_collection_task:
            self._metrics_collection_task.cancel()
        if self._adaptive_sizing_task:
            self._adaptive_sizing_task.cancel()
        
        # Cancel all waiting requests
        for scheduled_request in self._waiting_queue:
            scheduled_request.state = RequestState.CANCELLED
        
        self._waiting_queue.clear()
        self._running_requests.clear()
        
        self.logger.info("Enhanced Scheduler stopped")
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._is_running and not self._stop_event.is_set()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop()
