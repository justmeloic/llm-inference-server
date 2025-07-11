"""
API v1 endpoints for the LLM inference server.

This module defines the REST API endpoints for text generation,
health checks, and model information.
"""

import logging
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.status import (
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from ...core.config import get_settings
from ...schemas.request import GenerationRequest
from ...schemas.response import (
    ErrorResponse,
    GenerationResponse,
    HealthResponse,
    StreamingChunk,
)
from ...services.inference_service import InferenceService, get_inference_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_inference_service_dep() -> InferenceService:
    """Dependency to get the inference service"""
    return get_inference_service()


@router.get("/health", response_model=HealthResponse)
async def health_check(service: InferenceService = Depends(get_inference_service_dep)):
    """
    Health check endpoint.

    Returns the current status of the inference service and model.
    """
    try:
        settings = get_settings()

        return HealthResponse(
            status="healthy" if service.is_ready else "loading",
            model_loaded=service.is_ready,
            model_path=settings.model_path,
            memory_usage=None,  # Could add memory stats here
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}",
        )


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    service: InferenceService = Depends(get_inference_service_dep),
):
    """
    Generate text completion for a given prompt.

    This endpoint supports both regular and streaming responses based on
    the 'stream' parameter in the request.
    """
    if not service.is_ready:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service is not ready. Please wait for model to load.",
        )

    try:
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                _generate_stream(service, request),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8",
                },
            )
        else:
            # Return regular response
            result = await service.generate(request)
            return result

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


async def _generate_stream(service: InferenceService, request: GenerationRequest):
    """
    Generate streaming response.

    This function handles server-sent events for streaming text generation.
    """
    try:
        async for chunk in service.generate_stream(request):
            # Format as server-sent event
            chunk_text = chunk.get("choices", [{}])[0].get("text", "")

            # Only yield non-empty chunks
            if chunk_text:
                yield f"data: {chunk_text}\n\n"

            # Check if generation is complete
            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason")
            if finish_reason:
                yield f"data: [DONE]\n\n"
                break

    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        yield f"data: Error: {str(e)}\n\n"
        yield f"data: [DONE]\n\n"


@router.get("/models")
async def list_models(service: InferenceService = Depends(get_inference_service_dep)):
    """
    List available models.

    Returns information about the currently loaded model.
    """
    try:
        if not service.is_ready:
            return {"models": []}

        # Get model info from engine
        model_info = service.engine.get_model_info() if service.engine else {}

        return {
            "models": [
                {
                    "id": model_info.get("model_name", "unknown"),
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                    "path": model_info.get("model_path", ""),
                    "loaded": model_info.get("loaded", False),
                    "context_size": model_info.get("n_ctx", 0),
                    "gpu_layers": model_info.get("n_gpu_layers", 0),
                }
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}",
        )


@router.get("/stats")
async def get_stats(service: InferenceService = Depends(get_inference_service_dep)):
    """
    Get inference service statistics.

    Returns performance and usage statistics.
    """
    try:
        settings = get_settings()

        stats = {
            "service_ready": service.is_ready,
            "model_loaded": service.is_ready,
            "queue_size": service.request_queue.qsize() if service.request_queue else 0,
            "batch_settings": {
                "max_batch_size": settings.max_batch_size,
                "batch_timeout": settings.batch_timeout,
            },
            "model_settings": {
                "n_ctx": settings.n_ctx,
                "n_gpu_layers": settings.n_gpu_layers,
                "n_batch": settings.n_batch,
            },
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )
