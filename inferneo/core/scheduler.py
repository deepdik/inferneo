"""
Advanced request scheduler with priority queuing and intelligent batching
"""

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from .config import SchedulerConfig


class RequestState(Enum):
    """Request states"""
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledRequest:
    """Request with scheduling metadata"""
    request_id: str
    priority: int
    created_at: float
    estimated_tokens: int
    state: RequestState = RequestState.WAITING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def __lt__(self, other):
        """Priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class Scheduler:
    """
    Advanced request scheduler with priority queuing, intelligent batching,
    and preemption support
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = None  # Will be set by engine
        
        # Request queues
        self._waiting_queue: List[ScheduledRequest] = []
        self._running_requests: Dict[str, ScheduledRequest] = {}
        self._completed_requests: Dict[str, ScheduledRequest] = {}
        
        # Batching state
        self._current_batch: List[ScheduledRequest] = []
        self._batch_tokens = 0
        self._max_batch_tokens = config.max_waiting_tokens
        
        # Performance tracking
        self._total_requests = 0
        self._total_wait_time = 0.0
        self._total_processing_time = 0.0
        
        # Async state
        self._is_running = False
        self._batch_ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
    
    def initialize(self):
        """Initialize the scheduler"""
        self._is_running = True
        self.logger.info("Scheduler initialized")
    
    async def add_request(self, request: Any) -> str:
        """
        Add a request to the scheduler
        
        Args:
            request: GenerationRequest object
            
        Returns:
            Request ID
        """
        # Estimate tokens for this request
        estimated_tokens = self._estimate_tokens(request)
        
        # Create scheduled request
        scheduled_request = ScheduledRequest(
            request_id=request.request_id,
            priority=request.priority,
            created_at=time.time(),
            estimated_tokens=estimated_tokens
        )
        
        # Add to waiting queue
        heapq.heappush(self._waiting_queue, scheduled_request)
        self._total_requests += 1
        
        # Check if we can form a batch
        await self._try_form_batch()
        
        return request.request_id
    
    async def get_next_batch(self) -> List[Any]:
        """
        Get the next batch of requests to process
        
        Returns:
            List of requests to process
        """
        if not self._current_batch:
            # Wait for batch to be ready or timeout
            try:
                await asyncio.wait_for(
                    self._batch_ready_event.wait(),
                    timeout=0.1  # 100ms timeout
                )
            except asyncio.TimeoutError:
                return []
        
        # Return current batch and clear it
        batch = self._current_batch.copy()
        self._current_batch.clear()
        self._batch_tokens = 0
        self._batch_ready_event.clear()
        
        # Move requests to running state
        for scheduled_request in batch:
            scheduled_request.state = RequestState.RUNNING
            scheduled_request.start_time = time.time()
            self._running_requests[scheduled_request.request_id] = scheduled_request
        
        return batch
    
    async def complete_request(self, request_id: str, success: bool = True):
        """
        Mark a request as completed
        
        Args:
            request_id: Request ID
            success: Whether the request completed successfully
        """
        if request_id not in self._running_requests:
            return
        
        scheduled_request = self._running_requests.pop(request_id)
        scheduled_request.end_time = time.time()
        
        if success:
            scheduled_request.state = RequestState.COMPLETED
            self._completed_requests[request_id] = scheduled_request
        else:
            scheduled_request.state = RequestState.FAILED
        
        # Update statistics
        if scheduled_request.start_time:
            processing_time = scheduled_request.end_time - scheduled_request.start_time
            self._total_processing_time += processing_time
    
    async def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request
        
        Args:
            request_id: Request ID
            
        Returns:
            True if request was cancelled, False if not found
        """
        # Check running requests
        if request_id in self._running_requests:
            scheduled_request = self._running_requests.pop(request_id)
            scheduled_request.state = RequestState.CANCELLED
            return True
        
        # Check waiting queue
        for i, scheduled_request in enumerate(self._waiting_queue):
            if scheduled_request.request_id == request_id:
                self._waiting_queue.pop(i)
                heapq.heapify(self._waiting_queue)  # Re-heapify
                scheduled_request.state = RequestState.CANCELLED
                return True
        
        return False
    
    async def _try_form_batch(self):
        """Try to form a batch from waiting requests"""
        if not self._waiting_queue:
            return
        
        # Check if we can add more requests to current batch
        while (self._waiting_queue and 
               self._batch_tokens < self._max_batch_tokens and
               len(self._current_batch) < self.config.max_num_partial_prefills):
            
            # Get highest priority request
            scheduled_request = heapq.heappop(self._waiting_queue)
            
            # Check if adding this request would exceed batch limits
            if (self._batch_tokens + scheduled_request.estimated_tokens > 
                self._max_batch_tokens):
                # Put it back and stop
                heapq.heappush(self._waiting_queue, scheduled_request)
                break
            
            # Add to current batch
            self._current_batch.append(scheduled_request)
            self._batch_tokens += scheduled_request.estimated_tokens
        
        # Signal that batch is ready
        if self._current_batch:
            self._batch_ready_event.set()
    
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
    
    def _preempt_low_priority_requests(self, high_priority_request: ScheduledRequest):
        """
        Preempt low priority requests to make room for high priority ones
        
        Args:
            high_priority_request: High priority request that needs to be scheduled
        """
        if self.config.preemption_mode == "recompute":
            # Move running requests back to waiting queue
            for request_id, scheduled_request in list(self._running_requests.items()):
                if scheduled_request.priority < high_priority_request.priority:
                    # Reset state and move back to waiting
                    scheduled_request.state = RequestState.WAITING
                    scheduled_request.start_time = None
                    self._running_requests.pop(request_id)
                    heapq.heappush(self._waiting_queue, scheduled_request)
        
        elif self.config.preemption_mode == "swap":
            # TODO: Implement swap-based preemption
            # This would involve saving KV cache to CPU and restoring later
            pass
    
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
        }
    
    def get_queue_state(self) -> Dict[str, Any]:
        """Get detailed queue state for monitoring"""
        return {
            "waiting_queue": [
                {
                    "request_id": req.request_id,
                    "priority": req.priority,
                    "estimated_tokens": req.estimated_tokens,
                    "wait_time": time.time() - req.created_at
                }
                for req in self._waiting_queue
            ],
            "running_requests": [
                {
                    "request_id": req.request_id,
                    "priority": req.priority,
                    "processing_time": time.time() - req.start_time if req.start_time else 0
                }
                for req in self._running_requests.values()
            ]
        }
    
    async def stop(self):
        """Stop the scheduler"""
        self._is_running = False
        self._stop_event.set()
        
        # Cancel all waiting requests
        for scheduled_request in self._waiting_queue:
            scheduled_request.state = RequestState.CANCELLED
        
        self._waiting_queue.clear()
        self._running_requests.clear()
        
        self.logger.info("Scheduler stopped")
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._is_running and not self._stop_event.is_set() 