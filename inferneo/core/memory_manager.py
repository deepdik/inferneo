"""
Advanced memory manager with PagedAttention implementation
"""

import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import torch
import torch.nn as nn


class BlockState(Enum):
    """Memory block states"""
    FREE = "free"
    ALLOCATED = "allocated"
    SWAPPED = "swapped"
    RESERVED = "reserved"


@dataclass
class MemoryBlock:
    """Represents a memory block"""
    block_id: int
    start_offset: int
    size: int
    state: BlockState = BlockState.FREE
    owner: Optional[str] = None
    last_accessed: float = 0.0
    access_count: int = 0


class PagedAttention:
    """
    PagedAttention implementation for efficient memory management
    """
    
    def __init__(self, block_size: int = 16, max_blocks: int = 1000):
        self.block_size = block_size
        self.max_blocks = max_blocks
        
        # Memory blocks
        self.blocks: List[MemoryBlock] = []
        self.free_blocks: Set[int] = set()
        self.allocated_blocks: Dict[str, List[int]] = {}  # sequence_id -> block_ids
        
        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.total_swaps = 0
        
        # Initialize blocks
        self._init_blocks()
    
    def _init_blocks(self):
        """Initialize memory blocks"""
        for i in range(self.max_blocks):
            block = MemoryBlock(
                block_id=i,
                start_offset=i * self.block_size,
                size=self.block_size
            )
            self.blocks.append(block)
            self.free_blocks.add(i)
    
    def allocate_blocks(self, sequence_id: str, num_blocks: int) -> List[int]:
        """
        Allocate blocks for a sequence
        
        Args:
            sequence_id: Unique sequence identifier
            num_blocks: Number of blocks to allocate
            
        Returns:
            List of allocated block IDs
        """
        if len(self.free_blocks) < num_blocks:
            # Need to free some blocks
            self._evict_blocks(num_blocks - len(self.free_blocks))
        
        allocated_block_ids = []
        for _ in range(num_blocks):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available")
            
            block_id = self.free_blocks.pop()
            block = self.blocks[block_id]
            block.state = BlockState.ALLOCATED
            block.owner = sequence_id
            block.last_accessed = time.time()
            block.access_count = 1
            
            allocated_block_ids.append(block_id)
        
        self.allocated_blocks[sequence_id] = allocated_block_ids
        self.total_allocations += num_blocks
        
        return allocated_block_ids
    
    def free_blocks(self, sequence_id: str) -> int:
        """
        Free blocks for a sequence
        
        Args:
            sequence_id: Sequence identifier
            
        Returns:
            Number of freed blocks
        """
        if sequence_id not in self.allocated_blocks:
            return 0
        
        block_ids = self.allocated_blocks.pop(sequence_id)
        freed_count = 0
        
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.state = BlockState.FREE
            block.owner = None
            block.access_count = 0
            self.free_blocks.add(block_id)
            freed_count += 1
        
        self.total_deallocations += freed_count
        return freed_count
    
    def access_blocks(self, sequence_id: str, block_ids: List[int]):
        """
        Mark blocks as accessed (for LRU eviction)
        
        Args:
            sequence_id: Sequence identifier
            block_ids: Block IDs being accessed
        """
        current_time = time.time()
        for block_id in block_ids:
            if block_id < len(self.blocks):
                block = self.blocks[block_id]
                block.last_accessed = current_time
                block.access_count += 1
    
    def _evict_blocks(self, num_blocks_needed: int):
        """
        Evict blocks using LRU policy
        
        Args:
            num_blocks_needed: Number of blocks to evict
        """
        # Find least recently used blocks
        lru_blocks = []
        for sequence_id, block_ids in self.allocated_blocks.items():
            for block_id in block_ids:
                block = self.blocks[block_id]
                lru_blocks.append((block.last_accessed, sequence_id, block_id))
        
        # Sort by last accessed time
        lru_blocks.sort()
        
        # Evict oldest blocks
        evicted_count = 0
        for _, sequence_id, block_id in lru_blocks:
            if evicted_count >= num_blocks_needed:
                break
            
            # Free the block
            block = self.blocks[block_id]
            block.state = BlockState.FREE
            block.owner = None
            self.free_blocks.add(block_id)
            
            # Remove from allocated blocks
            if sequence_id in self.allocated_blocks:
                self.allocated_blocks[sequence_id].remove(block_id)
                if not self.allocated_blocks[sequence_id]:
                    del self.allocated_blocks[sequence_id]
            
            evicted_count += 1
        
        self.total_swaps += evicted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get PagedAttention statistics"""
        return {
            "total_blocks": self.max_blocks,
            "free_blocks": len(self.free_blocks),
            "allocated_blocks": sum(len(blocks) for blocks in self.allocated_blocks.values()),
            "total_allocations": self.total_allocations,
            "total_deallocations": self.total_deallocations,
            "total_swaps": self.total_swaps,
            "utilization": (self.max_blocks - len(self.free_blocks)) / self.max_blocks,
        }


class MemoryManager:
    """
    Advanced memory manager with PagedAttention and GPU memory optimization
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = None  # Will be set by engine
        
        # PagedAttention
        self.paged_attention = PagedAttention(
            block_size=config.block_size,
            max_blocks=config.max_num_seqs * 10  # Estimate
        )
        
        # GPU memory tracking
        self.gpu_memory_allocated = 0
        self.gpu_memory_reserved = 0
        self.max_gpu_memory = 0
        
        # Memory pools
        self.kv_cache_pool: Dict[str, torch.Tensor] = {}
        self.activation_pool: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.peak_memory_usage = 0
        
        # Initialize GPU memory tracking
        self._init_gpu_memory()
    
    def _init_gpu_memory(self):
        """Initialize GPU memory tracking"""
        if torch.cuda.is_available():
            self.max_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            self.gpu_memory_reserved = int(self.max_gpu_memory * self.config.gpu_memory_utilization)
        else:
            self.max_gpu_memory = 0
            self.gpu_memory_reserved = 0
    
    def initialize(self):
        """Initialize the memory manager"""
        self.logger.info("Memory manager initialized")
        self.logger.info(f"GPU memory: {self.max_gpu_memory / 1024**3:.2f} GB")
        self.logger.info(f"Reserved memory: {self.gpu_memory_reserved / 1024**3:.2f} GB")
    
    def allocate_kv_cache(self, sequence_id: str, num_layers: int, 
                         hidden_size: int, max_length: int) -> torch.Tensor:
        """
        Allocate KV cache for a sequence
        
        Args:
            sequence_id: Sequence identifier
            num_layers: Number of transformer layers
            hidden_size: Hidden size of the model
            max_length: Maximum sequence length
            
        Returns:
            KV cache tensor
        """
        # Calculate required memory
        cache_size = num_layers * 2 * hidden_size * max_length  # 2 for K and V
        cache_size_bytes = cache_size * 4  # Assuming float32
        
        # Check if we have enough memory
        if self.gpu_memory_allocated + cache_size_bytes > self.gpu_memory_reserved:
            self._evict_kv_cache(cache_size_bytes)
        
        # Allocate tensor
        kv_cache = torch.zeros(
            (num_layers, 2, max_length, hidden_size),
            dtype=torch.float16,  # Use float16 for memory efficiency
            device="cuda"
        )
        
        # Track allocation
        self.kv_cache_pool[sequence_id] = kv_cache
        self.gpu_memory_allocated += cache_size_bytes
        self.total_allocations += 1
        
        # Update peak usage
        self.peak_memory_usage = max(self.peak_memory_usage, self.gpu_memory_allocated)
        
        return kv_cache
    
    def free_kv_cache(self, sequence_id: str) -> bool:
        """
        Free KV cache for a sequence
        
        Args:
            sequence_id: Sequence identifier
            
        Returns:
            True if freed successfully
        """
        if sequence_id not in self.kv_cache_pool:
            return False
        
        kv_cache = self.kv_cache_pool.pop(sequence_id)
        cache_size_bytes = kv_cache.numel() * kv_cache.element_size()
        
        # Free tensor
        del kv_cache
        torch.cuda.empty_cache()
        
        # Update tracking
        self.gpu_memory_allocated -= cache_size_bytes
        self.total_deallocations += 1
        
        return True
    
    def allocate_activation_memory(self, sequence_id: str, 
                                 batch_size: int, seq_len: int, 
                                 hidden_size: int) -> torch.Tensor:
        """
        Allocate activation memory for forward pass
        
        Args:
            sequence_id: Sequence identifier
            batch_size: Batch size
            seq_len: Sequence length
            hidden_size: Hidden size
            
        Returns:
            Activation tensor
        """
        # Calculate required memory
        activation_size = batch_size * seq_len * hidden_size
        activation_size_bytes = activation_size * 4  # Assuming float32
        
        # Check if we have enough memory
        if self.gpu_memory_allocated + activation_size_bytes > self.gpu_memory_reserved:
            self._evict_activation_memory(activation_size_bytes)
        
        # Allocate tensor
        activation = torch.zeros(
            (batch_size, seq_len, hidden_size),
            dtype=torch.float16,
            device="cuda"
        )
        
        # Track allocation
        self.activation_pool[sequence_id] = activation
        self.gpu_memory_allocated += activation_size_bytes
        
        return activation
    
    def free_activation_memory(self, sequence_id: str) -> bool:
        """
        Free activation memory for a sequence
        
        Args:
            sequence_id: Sequence identifier
            
        Returns:
            True if freed successfully
        """
        if sequence_id not in self.activation_pool:
            return False
        
        activation = self.activation_pool.pop(sequence_id)
        activation_size_bytes = activation.numel() * activation.element_size()
        
        # Free tensor
        del activation
        torch.cuda.empty_cache()
        
        # Update tracking
        self.gpu_memory_allocated -= activation_size_bytes
        
        return True
    
    def _evict_kv_cache(self, required_bytes: int):
        """
        Evict KV cache to free memory
        
        Args:
            required_bytes: Bytes needed
        """
        # Simple LRU eviction - in practice, you'd want more sophisticated logic
        if not self.kv_cache_pool:
            return
        
        # Find oldest cache (simple heuristic)
        oldest_sequence = min(self.kv_cache_pool.keys())
        self.free_kv_cache(oldest_sequence)
    
    def _evict_activation_memory(self, required_bytes: int):
        """
        Evict activation memory to free memory
        
        Args:
            required_bytes: Bytes needed
        """
        if not self.activation_pool:
            return
        
        # Free oldest activation memory
        oldest_sequence = min(self.activation_pool.keys())
        self.free_activation_memory(oldest_sequence)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
        else:
            allocated = reserved = max_allocated = 0
        
        return {
            "gpu_memory_allocated": allocated,
            "gpu_memory_reserved": reserved,
            "gpu_memory_max_allocated": max_allocated,
            "gpu_memory_total": self.max_gpu_memory,
            "gpu_memory_utilization": allocated / self.max_gpu_memory if self.max_gpu_memory > 0 else 0,
            "kv_cache_entries": len(self.kv_cache_pool),
            "activation_entries": len(self.activation_pool),
            "total_allocations": self.total_allocations,
            "total_deallocations": self.total_deallocations,
            "peak_memory_usage": self.peak_memory_usage,
            "paged_attention_stats": self.paged_attention.get_stats(),
        }
    
    def optimize_memory(self):
        """Run memory optimization"""
        # Clear unused tensors
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        # Reset peak memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics"""
        return self.get_memory_stats() 