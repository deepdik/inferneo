"""
Multi-level cache manager for responses, KV cache, and model weights
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import asyncio

import torch
import torch.nn as nn


class CacheLevel(Enum):
    """Cache levels"""
    L1 = "l1"  # In-memory (fastest)
    L2 = "l2"  # GPU memory
    L3 = "l3"  # Disk cache (slowest)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    level: CacheLevel
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds


class CacheManager:
    """
    Multi-level cache manager with intelligent eviction and prefetching
    """
    
    def __init__(self, 
                 l1_size_mb: int = 1024,  # 1GB
                 l2_size_mb: int = 4096,  # 4GB
                 l3_size_mb: int = 10240,  # 10GB
                 default_ttl: int = 3600):  # 1 hour
        self.l1_size_mb = l1_size_mb
        self.l2_size_mb = l2_size_mb
        self.l3_size_mb = l3_size_mb
        self.default_ttl = default_ttl
        
        # Cache storage
        self.l1_cache: Dict[str, CacheEntry] = {}  # In-memory
        self.l2_cache: Dict[str, CacheEntry] = {}  # GPU memory
        self.l3_cache: Dict[str, CacheEntry] = {}  # Disk cache
        
        # Size tracking
        self.l1_size_bytes = 0
        self.l2_size_bytes = 0
        self.l3_size_bytes = 0
        
        # Statistics
        self.hits = {CacheLevel.L1: 0, CacheLevel.L2: 0, CacheLevel.L3: 0}
        self.misses = 0
        self.evictions = {CacheLevel.L1: 0, CacheLevel.L2: 0, CacheLevel.L3: 0}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def initialize(self):
        """Initialize the cache manager"""
        self._is_running = True
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while self._is_running:
                try:
                    await self._cleanup_expired_entries()
                    await asyncio.sleep(60)  # Run every minute
                except Exception as e:
                    print(f"Cache cleanup error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Try L1 cache first
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.hits[CacheLevel.L1] += 1
                return entry.value
        
        # Try L2 cache
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.hits[CacheLevel.L2] += 1
                # Promote to L1 if possible
                self._promote_to_l1(key, entry)
                return entry.value
        
        # Try L3 cache
        if key in self.l3_cache:
            entry = self.l3_cache[key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.hits[CacheLevel.L3] += 1
                # Promote to L2 if possible
                self._promote_to_l2(key, entry)
                return entry.value
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any, level: CacheLevel = CacheLevel.L1, 
            ttl: Optional[float] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            level: Cache level
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        # Calculate size
        size_bytes = self._estimate_size(value)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            level=level,
            created_at=time.time(),
            last_accessed=time.time(),
            size_bytes=size_bytes,
            ttl=ttl or self.default_ttl
        )
        
        # Try to cache at specified level
        if level == CacheLevel.L1:
            return self._set_l1(key, entry)
        elif level == CacheLevel.L2:
            return self._set_l2(key, entry)
        elif level == CacheLevel.L3:
            return self._set_l3(key, entry)
        
        return False
    
    def _set_l1(self, key: str, entry: CacheEntry) -> bool:
        """Set value in L1 cache"""
        # Check if we have space
        if self.l1_size_bytes + entry.size_bytes > self.l1_size_mb * 1024 * 1024:
            # Need to evict some entries
            self._evict_l1(entry.size_bytes)
        
        # Add to cache
        self.l1_cache[key] = entry
        self.l1_size_bytes += entry.size_bytes
        return True
    
    def _set_l2(self, key: str, entry: CacheEntry) -> bool:
        """Set value in L2 cache"""
        # Check if we have space
        if self.l2_size_bytes + entry.size_bytes > self.l2_size_mb * 1024 * 1024:
            # Need to evict some entries
            self._evict_l2(entry.size_bytes)
        
        # Add to cache
        self.l2_cache[key] = entry
        self.l2_size_bytes += entry.size_bytes
        return True
    
    def _set_l3(self, key: str, entry: CacheEntry) -> bool:
        """Set value in L3 cache"""
        # Check if we have space
        if self.l3_size_bytes + entry.size_bytes > self.l3_size_mb * 1024 * 1024:
            # Need to evict some entries
            self._evict_l3(entry.size_bytes)
        
        # Add to cache
        self.l3_cache[key] = entry
        self.l3_size_bytes += entry.size_bytes
        return True
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1"""
        if self._set_l1(key, entry):
            # Remove from L2
            self.l2_cache.pop(key)
            self.l2_size_bytes -= entry.size_bytes
    
    def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry from L3 to L2"""
        if self._set_l2(key, entry):
            # Remove from L3
            self.l3_cache.pop(key)
            self.l3_size_bytes -= entry.size_bytes
    
    def _evict_l1(self, required_bytes: int):
        """Evict entries from L1 cache"""
        # Sort by access time (LRU)
        entries = sorted(
            self.l1_cache.values(),
            key=lambda e: e.last_accessed
        )
        
        freed_bytes = 0
        for entry in entries:
            if freed_bytes >= required_bytes:
                break
            
            self.l1_cache.pop(entry.key)
            self.l1_size_bytes -= entry.size_bytes
            freed_bytes += entry.size_bytes
            self.evictions[CacheLevel.L1] += 1
    
    def _evict_l2(self, required_bytes: int):
        """Evict entries from L2 cache"""
        # Sort by access time (LRU)
        entries = sorted(
            self.l2_cache.values(),
            key=lambda e: e.last_accessed
        )
        
        freed_bytes = 0
        for entry in entries:
            if freed_bytes >= required_bytes:
                break
            
            self.l2_cache.pop(entry.key)
            self.l2_size_bytes -= entry.size_bytes
            freed_bytes += entry.size_bytes
            self.evictions[CacheLevel.L2] += 1
    
    def _evict_l3(self, required_bytes: int):
        """Evict entries from L3 cache"""
        # Sort by access time (LRU)
        entries = sorted(
            self.l3_cache.values(),
            key=lambda e: e.last_accessed
        )
        
        freed_bytes = 0
        for entry in entries:
            if freed_bytes >= required_bytes:
                break
            
            self.l3_cache.pop(entry.key)
            self.l3_size_bytes -= entry.size_bytes
            freed_bytes += entry.size_bytes
            self.evictions[CacheLevel.L3] += 1
    
    def _update_access(self, entry: CacheEntry):
        """Update access metadata"""
        entry.last_accessed = time.time()
        entry.access_count += 1
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired"""
        if entry.ttl is None:
            return False
        return time.time() - entry.created_at > entry.ttl
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(v) for v in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(v) for v in value.values())
        else:
            # Rough estimate for other types
            return 1024  # 1KB default
    
    async def _cleanup_expired_entries(self):
        """Remove expired entries from all cache levels"""
        current_time = time.time()
        
        # Clean L1
        expired_keys = [
            key for key, entry in self.l1_cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            entry = self.l1_cache.pop(key)
            self.l1_size_bytes -= entry.size_bytes
        
        # Clean L2
        expired_keys = [
            key for key, entry in self.l2_cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            entry = self.l2_cache.pop(key)
            self.l2_size_bytes -= entry.size_bytes
        
        # Clean L3
        expired_keys = [
            key for key, entry in self.l3_cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            entry = self.l3_cache.pop(key)
            self.l3_size_bytes -= entry.size_bytes
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate cache entry
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was found and invalidated
        """
        found = False
        
        if key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self.l1_size_bytes -= entry.size_bytes
            found = True
        
        if key in self.l2_cache:
            entry = self.l2_cache.pop(key)
            self.l2_size_bytes -= entry.size_bytes
            found = True
        
        if key in self.l3_cache:
            entry = self.l3_cache.pop(key)
            self.l3_size_bytes -= entry.size_bytes
            found = True
        
        return found
    
    def clear(self, level: Optional[CacheLevel] = None):
        """
        Clear cache
        
        Args:
            level: Cache level to clear, or None for all levels
        """
        if level is None or level == CacheLevel.L1:
            self.l1_cache.clear()
            self.l1_size_bytes = 0
        
        if level is None or level == CacheLevel.L2:
            self.l2_cache.clear()
            self.l2_size_bytes = 0
        
        if level is None or level == CacheLevel.L3:
            self.l3_cache.clear()
            self.l3_size_bytes = 0
    
    def prefetch(self, keys: List[str], level: CacheLevel = CacheLevel.L1):
        """
        Prefetch keys into cache
        
        Args:
            keys: List of keys to prefetch
            level: Target cache level
        """
        # This is a placeholder - in practice, you'd implement
        # actual prefetching logic based on access patterns
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(self.hits.values())
        total_requests = total_hits + self.misses
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "l1_entries": len(self.l1_cache),
            "l1_size_mb": self.l1_size_bytes / (1024 * 1024),
            "l1_hits": self.hits[CacheLevel.L1],
            "l1_evictions": self.evictions[CacheLevel.L1],
            
            "l2_entries": len(self.l2_cache),
            "l2_size_mb": self.l2_size_bytes / (1024 * 1024),
            "l2_hits": self.hits[CacheLevel.L2],
            "l2_evictions": self.evictions[CacheLevel.L2],
            
            "l3_entries": len(self.l3_cache),
            "l3_size_mb": self.l3_size_bytes / (1024 * 1024),
            "l3_hits": self.hits[CacheLevel.L3],
            "l3_evictions": self.evictions[CacheLevel.L3],
            
            "total_hits": total_hits,
            "total_misses": self.misses,
            "hit_rate": hit_rate,
            "total_evictions": sum(self.evictions.values()),
        }
    
    def stop(self):
        """Stop the cache manager"""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
    
    async def cleanup(self):
        pass 