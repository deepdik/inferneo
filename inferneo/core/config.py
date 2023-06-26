"""
Configuration system for Turbo Inference Server
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import torch
from pydantic import BaseModel, Field


class QuantizationConfig(BaseModel):
    """Configuration for model quantization"""
    method: str = "none"  # "none", "awq", "gptq", "int8", "fp8"
    bits: int = 16
    group_size: int = 128
    zero_point: bool = True
    scale_method: str = "max"
    param_path: Optional[str] = None


class MemoryConfig(BaseModel):
    """Configuration for memory management"""
    gpu_memory_utilization: float = 0.9
    cpu_offload: bool = False
    swap_space: int = 4  # GB
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    block_size: int = 16
    enable_prefix_caching: bool = True
    enable_kv_cache: bool = True


class SchedulerConfig(BaseModel):
    """Configuration for request scheduling"""
    enable_chunked_prefill: bool = True
    max_num_partial_prefills: int = 2
    long_prefill_token_threshold: int = 8192
    preemption_mode: str = "recompute"  # "recompute", "swap"
    max_waiting_tokens: int = 2048
    enable_priority_queue: bool = True
    max_priority_levels: int = 10


class ParallelConfig(BaseModel):
    """Configuration for distributed inference"""
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_sequence_parallelism: bool = False
    enable_async_tp: bool = False


class SecurityConfig(BaseModel):
    """Configuration for security features"""
    api_key: Optional[str] = None
    rate_limit: int = 1000  # requests per minute
    max_request_size: str = "10MB"
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_auth: bool = False
    jwt_secret: Optional[str] = None


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability"""
    metrics_port: int = 9090
    health_check_interval: int = 30
    log_level: str = "INFO"
    enable_tracing: bool = True
    enable_profiling: bool = False
    prometheus_endpoint: str = "/metrics"


@dataclass
class EngineConfig:
    """Main configuration for the Turbo Engine"""
    
    # Model settings
    model: str
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    dtype: Union[str, torch.dtype] = "auto"
    seed: Optional[int] = None
    
    # Performance settings
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_paddings: int = 256
    
    # Memory settings
    gpu_memory_utilization: float = 0.9
    swap_space: int = 4  # GB
    cpu_offload: bool = False
    
    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Parallel settings
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    
    # Scheduling
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Memory management
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Advanced settings
    enable_cuda_graph: bool = True
    enable_flash_attention: bool = True
    enable_xformers: bool = True
    enable_speculative_decoding: bool = False
    speculative_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if self.tokenizer is None:
            self.tokenizer = self.model
            
        if self.dtype == "auto":
            self.dtype = torch.float16
            
        # Set memory config from engine config
        self.memory.max_model_len = self.max_model_len
        self.memory.max_num_seqs = self.max_num_seqs
        self.memory.max_num_batched_tokens = self.max_num_batched_tokens
        self.memory.gpu_memory_utilization = self.gpu_memory_utilization
        self.memory.cpu_offload = self.cpu_offload
        self.memory.swap_space = self.swap_space
        
        # Set scheduler config from engine config
        self.scheduler.max_num_partial_prefills = min(
            self.scheduler.max_num_partial_prefills, 
            self.max_num_seqs // 2
        )


@dataclass
class ServerConfig:
    """Configuration for the HTTP/WebSocket server"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Security
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # API settings
    enable_openai_compatibility: bool = True
    enable_websocket: bool = True
    enable_streaming: bool = True
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_window: int = 60  # seconds
    rate_limit_max_requests: int = 1000
    
    # Caching
    enable_response_cache: bool = True
    cache_ttl: int = 3600  # seconds
    cache_max_size: int = 10000
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.workers > 1 and self.reload:
            raise ValueError("Cannot use reload=True with multiple workers")
            
        if self.security.rate_limit > 0:
            self.rate_limit_max_requests = self.security.rate_limit


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, config_type: str = "engine") -> Union[EngineConfig, ServerConfig]:
        """Load configuration from file or environment"""
        if config_type == "engine":
            return self._load_engine_config()
        elif config_type == "server":
            return self._load_server_config()
        else:
            raise ValueError(f"Unknown config type: {config_type}")
    
    def _load_engine_config(self) -> EngineConfig:
        """Load engine configuration"""
        # Load from environment variables
        config = EngineConfig(
            model=os.getenv("TURBO_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            max_model_len=int(os.getenv("TURBO_MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.getenv("TURBO_MAX_NUM_SEQS", "256")),
            max_num_batched_tokens=int(os.getenv("TURBO_MAX_BATCHED_TOKENS", "8192")),
            gpu_memory_utilization=float(os.getenv("TURBO_GPU_MEMORY_UTIL", "0.9")),
            quantization=QuantizationConfig(
                method=os.getenv("TURBO_QUANTIZATION", "none")
            )
        )
        
        return config
    
    def _load_server_config(self) -> ServerConfig:
        """Load server configuration"""
        config = ServerConfig(
            host=os.getenv("TURBO_HOST", "0.0.0.0"),
            port=int(os.getenv("TURBO_PORT", "8000")),
            workers=int(os.getenv("TURBO_WORKERS", "1")),
            security=SecurityConfig(
                api_key=os.getenv("TURBO_API_KEY"),
                rate_limit=int(os.getenv("TURBO_RATE_LIMIT", "1000"))
            ),
            monitoring=MonitoringConfig(
                metrics_port=int(os.getenv("TURBO_METRICS_PORT", "9090")),
                log_level=os.getenv("TURBO_LOG_LEVEL", "INFO")
            )
        )
        
        return config
    
    def save_config(self, config: Union[EngineConfig, ServerConfig], 
                   config_path: Optional[str] = None) -> None:
        """Save configuration to file"""
        import yaml
        
        config_path = config_path or self.config_path
        if not config_path:
            raise ValueError("No config path specified")
            
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def _config_to_dict(self, config: Union[EngineConfig, ServerConfig]) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        if isinstance(config, EngineConfig):
            return {
                "model": config.model,
                "tokenizer": config.tokenizer,
                "max_model_len": config.max_model_len,
                "max_num_seqs": config.max_num_seqs,
                "max_num_batched_tokens": config.max_num_batched_tokens,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "quantization": config.quantization.dict(),
                "parallel": config.parallel.dict(),
                "scheduler": config.scheduler.dict(),
                "memory": config.memory.dict(),
            }
        elif isinstance(config, ServerConfig):
            return {
                "host": config.host,
                "port": config.port,
                "workers": config.workers,
                "security": config.security.dict(),
                "monitoring": config.monitoring.dict(),
            }
        else:
            raise ValueError(f"Unsupported config type: {type(config)}") 