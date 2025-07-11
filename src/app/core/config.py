"""Application configuration using Pydantic Settings"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, description="Server port")

    # Model Configuration
    model_path: str = Field(
        default="./models/phi-3-mini-4k-instruct-q4.gguf",
        description="Path to the GGUF model file",
    )

    # llama-cpp-python Configuration
    n_gpu_layers: int = Field(
        default=32, description="Number of layers to offload to GPU (-1 for all)"
    )
    n_ctx: int = Field(default=4096, description="Context window size")
    n_batch: int = Field(default=512, description="Batch size for prompt processing")
    n_threads: Optional[int] = Field(
        default=None, description="Number of threads (None for auto-detect)"
    )
    use_mlock: bool = Field(
        default=True, description="Use mlock to prevent model from being swapped"
    )
    use_mmap: bool = Field(
        default=True, description="Use memory mapping for model loading"
    )

    # Dynamic Batching Configuration
    max_batch_size: int = Field(
        default=8, description="Maximum number of requests to batch together"
    )
    batch_timeout: float = Field(
        default=0.1, description="Maximum time to wait for batch formation (seconds)"
    )

    # Generation Defaults
    default_max_tokens: int = Field(
        default=256, description="Default maximum tokens to generate"
    )
    default_temperature: float = Field(
        default=0.7, description="Default temperature for generation"
    )
    default_top_p: float = Field(
        default=0.9, description="Default top-p for nucleus sampling"
    )
    default_top_k: int = Field(default=40, description="Default top-k for sampling")

    # Performance Configuration
    verbose: bool = Field(
        default=False, description="Enable verbose logging from llama-cpp"
    )

    def validate_model_path(self) -> bool:
        """Validate that the model path exists"""
        return os.path.exists(self.model_path)

    @property
    def model_exists(self) -> bool:
        """Check if model file exists"""
        return self.validate_model_path()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
