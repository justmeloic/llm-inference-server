"""Response schemas for API endpoints"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerationUsage(BaseModel):
    """Token usage statistics"""

    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total number of tokens used")


class GenerationChoice(BaseModel):
    """A single generation choice"""

    text: str = Field(description="The generated text")
    finish_reason: str = Field(description="Reason for stopping generation")
    index: int = Field(description="Index of this choice")


class GenerationResponse(BaseModel):
    """Response model for text generation"""

    id: str = Field(description="Unique identifier for this generation")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(
        description="Unix timestamp of when the generation was created"
    )
    model: str = Field(description="Model used for generation")
    choices: List[GenerationChoice] = Field(description="List of generated choices")
    usage: GenerationUsage = Field(description="Token usage statistics")


class StreamingChunk(BaseModel):
    """A single chunk in a streaming response"""

    id: str = Field(description="Unique identifier for this generation")
    object: str = Field(default="text_completion", description="Object type")
    created: int = Field(
        description="Unix timestamp of when the generation was created"
    )
    model: str = Field(description="Model used for generation")
    choices: List[Dict[str, Any]] = Field(description="List of streaming choices")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(default="healthy", description="Health status")
    model_loaded: bool = Field(description="Whether the model is loaded")
    model_path: str = Field(description="Path to the loaded model")
    memory_usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Memory usage statistics"
    )


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(
        default=None, description="Detailed error information"
    )
    code: Optional[int] = Field(default=None, description="Error code")
