"""Request schemas for API endpoints"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """Request model for text generation"""

    prompt: str = Field(
        ...,
        description="The input prompt for text generation",
        min_length=1,
        max_length=8192,
    )

    max_tokens: int = Field(
        default=256, description="Maximum number of tokens to generate", ge=1, le=2048
    )

    temperature: float = Field(
        default=0.7, description="Sampling temperature (0.0 to 2.0)", ge=0.0, le=2.0
    )

    top_p: float = Field(
        default=0.9, description="Top-p nucleus sampling parameter", ge=0.0, le=1.0
    )

    top_k: int = Field(default=40, description="Top-k sampling parameter", ge=1, le=100)

    stream: bool = Field(default=False, description="Whether to stream the response")

    stop: Optional[list[str]] = Field(
        default=None, description="List of stop sequences"
    )

    repeat_penalty: float = Field(
        default=1.1, description="Penalty for token repetition", ge=0.0, le=2.0
    )

    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "What is the capital of France?",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "stream": False,
                "stop": ["</s>", "\n\n"],
                "repeat_penalty": 1.1,
                "seed": 42,
            }
        }
