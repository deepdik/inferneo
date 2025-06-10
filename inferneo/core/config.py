"""
Configuration system for Inferneo
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

import torch
from pydantic import BaseModel, Field


class QuantizationMethod(Enum):
    """Available quantization methods"""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"


class SchedulerType(Enum):
    """Available scheduler types"""
    FIFO = "fifo"
    PRIORITY = "priority"
    FAIR = "fair"


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
class ModelConfig:
    """Configuration for model loading and inference"""
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    quantization: QuantizationMethod = QuantizationMethod.NONE
    trust_remote_code: bool = True
    revision: Optional[str] = None
    tokenizer: Optional[str] = None


@dataclass
class ServerConfig:
    """Configuration for the HTTP/gRPC server"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    enable_cors: bool = True
    api_key: Optional[str] = None
    rate_limit: int = 1000


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and metrics"""
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_prometheus: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30


@dataclass
class EngineConfig:
    """Main configuration for the Inferneo Engine"""
    
    # Model configuration
    model: str = "meta-llama/Llama-2-7b-chat-hf"
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    quantization: QuantizationMethod = QuantizationMethod.NONE
    trust_remote_code: bool = True
    revision: Optional[str] = None
    tokenizer: Optional[str] = None
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    enable_cors: bool = True
    api_key: Optional[str] = None
    rate_limit: int = 1000
    
    # Monitoring configuration
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_prometheus: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30
    
    # Performance settings
    enable_cuda_graph: bool = True
    enable_paged_attention: bool = True
    enable_speculative_decoding: bool = True
    enable_kv_cache: bool = True
    max_workers: int = 4
    
    # Memory settings
    max_memory_gb: float = 16.0
    memory_fraction: float = 0.9
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    # Scheduler settings
    scheduler_type: SchedulerType = SchedulerType.FAIR
    max_batch_size: int = 32
    max_wait_time: float = 0.1
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        # Model settings
        self.model = os.getenv("INFERNEO_MODEL", self.model)
        self.max_model_len = int(os.getenv("INFERNEO_MAX_MODEL_LEN", str(self.max_model_len)))
        self.max_num_seqs = int(os.getenv("INFERNEO_MAX_NUM_SEQS", str(self.max_num_seqs)))
        self.max_num_batched_tokens = int(os.getenv("INFERNEO_MAX_BATCHED_TOKENS", str(self.max_num_batched_tokens)))
        self.gpu_memory_utilization = float(os.getenv("INFERNEO_GPU_MEMORY_UTIL", str(self.gpu_memory_utilization)))
        self.quantization = QuantizationMethod(os.getenv("INFERNEO_QUANTIZATION", self.quantization.value))
        
        # Server settings
        self.host = os.getenv("INFERNEO_HOST", self.host)
        self.port = int(os.getenv("INFERNEO_PORT", str(self.port)))
        self.workers = int(os.getenv("INFERNEO_WORKERS", str(self.workers)))
        self.api_key = os.getenv("INFERNEO_API_KEY", self.api_key)
        self.rate_limit = int(os.getenv("INFERNEO_RATE_LIMIT", str(self.rate_limit)))
        
        # Monitoring settings
        self.metrics_port = int(os.getenv("INFERNEO_METRICS_PORT", str(self.metrics_port)))
        self.log_level = os.getenv("INFERNEO_LOG_LEVEL", self.log_level)


def load_config_from_file(config_path: str) -> EngineConfig:
    """Load configuration from a JSON or YAML file"""
    import json
    import yaml
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            config_data = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported config file format. Use .json or .yaml")
    
    return EngineConfig(**config_data)


def save_config_to_file(config: EngineConfig, config_path: str):
    """Save configuration to a JSON or YAML file"""
    import json
    import yaml
    
    config_dict = {
        "model": config.model,
        "max_model_len": config.max_model_len,
        "max_num_seqs": config.max_num_seqs,
        "max_num_batched_tokens": config.max_num_batched_tokens,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "quantization": config.quantization.value,
        "host": config.host,
        "port": config.port,
        "workers": config.workers,
        "rate_limit": config.rate_limit,
        "metrics_port": config.metrics_port,
        "log_level": config.log_level,
        "enable_cuda_graph": config.enable_cuda_graph,
        "enable_paged_attention": config.enable_paged_attention,
        "enable_speculative_decoding": config.enable_speculative_decoding,
        "max_workers": config.max_workers,
        "max_memory_gb": config.max_memory_gb,
        "cache_size": config.cache_size,
        "cache_ttl": config.cache_ttl,
        "scheduler_type": config.scheduler_type.value,
        "max_batch_size": config.max_batch_size,
        "max_wait_time": config.max_wait_time,
    }
    
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config_dict, f, indent=2)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported config file format. Use .json or .yaml")


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
            model=os.getenv("INFERNEO_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
            max_model_len=int(os.getenv("INFERNEO_MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.getenv("INFERNEO_MAX_NUM_SEQS", "256")),
            max_num_batched_tokens=int(os.getenv("INFERNEO_MAX_BATCHED_TOKENS", "8192")),
            gpu_memory_utilization=float(os.getenv("INFERNEO_GPU_MEMORY_UTIL", "0.9")),
            quantization=QuantizationConfig(
                method=os.getenv("INFERNEO_QUANTIZATION", "none")
            )
        )
        
        return config
    
    def _load_server_config(self) -> ServerConfig:
        """Load server configuration"""
        config = ServerConfig(
            host=os.getenv("INFERNEO_HOST", "0.0.0.0"),
            port=int(os.getenv("INFERNEO_PORT", "8000")),
            workers=int(os.getenv("INFERNEO_WORKERS", "1")),
            security=SecurityConfig(
                api_key=os.getenv("INFERNEO_API_KEY"),
                rate_limit=int(os.getenv("INFERNEO_RATE_LIMIT", "1000"))
            ),
            monitoring=MonitoringConfig(
                metrics_port=int(os.getenv("INFERNEO_METRICS_PORT", "9090")),
                log_level=os.getenv("INFERNEO_LOG_LEVEL", "INFO")
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