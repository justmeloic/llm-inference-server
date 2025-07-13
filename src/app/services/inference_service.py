"""Inference service with dynamic batching"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...inference.engine import InferenceEngine
from ...lib.core.config import get_settings
from ..schemas.request import GenerationRequest

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A batched request with metadata"""

    request_id: str
    request: GenerationRequest
    future: asyncio.Future
    timestamp: float


class InferenceService:
    """
    Inference service that implements dynamic batching for improved throughput.

    This service maintains a queue of incoming requests and processes them in batches
    to optimize GPU utilization and reduce per-request latency.
    """

    def __init__(self):
        self.settings = get_settings()
        self.engine: Optional[InferenceEngine] = None
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.batch_processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the inference service"""
        async with self._lock:
            if self.engine is None:
                logger.info("Initializing inference engine...")
                self.engine = InferenceEngine(self.settings)
                await self.engine.initialize()
                logger.info("Inference engine initialized successfully")

            if not self.is_running:
                self.is_running = True
                self.batch_processor_task = asyncio.create_task(self._batch_processor())
                logger.info("Batch processor started")

    async def shutdown(self):
        """Shutdown the inference service"""
        async with self._lock:
            if self.is_running:
                self.is_running = False
                if self.batch_processor_task:
                    self.batch_processor_task.cancel()
                    try:
                        await self.batch_processor_task
                    except asyncio.CancelledError:
                        pass
                logger.info("Batch processor stopped")

            if self.engine:
                await self.engine.shutdown()
                self.engine = None
                logger.info("Inference engine shutdown")

    async def generate(self, request: GenerationRequest) -> Dict[str, Any]:
        """
        Generate text for a single request using dynamic batching.

        Args:
            request: The generation request

        Returns:
            Generated response dictionary
        """
        if not self.is_running or self.engine is None:
            raise RuntimeError("Inference service not initialized")

        # Create a future to wait for the result
        future = asyncio.Future()
        request_id = str(uuid.uuid4())

        # Create batch request
        batch_request = BatchRequest(
            request_id=request_id, request=request, future=future, timestamp=time.time()
        )

        # Add to queue
        await self.request_queue.put(batch_request)

        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise

    async def _batch_processor(self):
        """
        Background task that processes requests in batches.

        This task continuously pulls requests from the queue and forms batches
        based on the configured batch size and timeout.
        """
        logger.info("Batch processor started")

        while self.is_running:
            try:
                # Collect batch of requests
                batch_requests = await self._collect_batch()

                if not batch_requests:
                    continue

                # Process the batch
                await self._process_batch(batch_requests)

            except asyncio.CancelledError:
                logger.info("Batch processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Small delay to prevent tight error loops
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[BatchRequest]:
        """
        Collect a batch of requests from the queue.

        Returns:
            List of batch requests to process
        """
        batch_requests = []
        batch_start_time = time.time()

        # Wait for first request
        try:
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=1.0,  # 1 second timeout to check if we should shutdown
            )
            batch_requests.append(first_request)
        except asyncio.TimeoutError:
            return []

        # Collect additional requests until batch is full or timeout
        while (
            len(batch_requests) < self.settings.max_batch_size
            and time.time() - batch_start_time < self.settings.batch_timeout
        ):
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=max(
                        0.01,
                        self.settings.batch_timeout - (time.time() - batch_start_time),
                    ),
                )
                batch_requests.append(request)
            except asyncio.TimeoutError:
                break

        return batch_requests

    async def _process_batch(self, batch_requests: List[BatchRequest]):
        """
        Process a batch of requests.

        Args:
            batch_requests: List of requests to process
        """
        if not batch_requests:
            return

        logger.debug(f"Processing batch of {len(batch_requests)} requests")

        try:
            # Extract prompts and parameters
            prompts = [req.request.prompt for req in batch_requests]

            # For now, use the first request's parameters for the entire batch
            # In a more sophisticated implementation, you could group by similar parameters
            first_request = batch_requests[0].request

            # Generate responses for the batch
            responses = await self.engine.generate_batch(
                prompts=prompts,
                max_tokens=first_request.max_tokens,
                temperature=first_request.temperature,
                top_p=first_request.top_p,
                top_k=first_request.top_k,
                stop=first_request.stop,
                repeat_penalty=first_request.repeat_penalty,
                seed=first_request.seed,
            )

            # Send results back to waiting requests
            for i, batch_request in enumerate(batch_requests):
                try:
                    if i < len(responses):
                        batch_request.future.set_result(responses[i])
                    else:
                        batch_request.future.set_exception(
                            RuntimeError(
                                f"No response for request {batch_request.request_id}"
                            )
                        )
                except Exception as e:
                    logger.error(
                        f"Error setting result for request {batch_request.request_id}: {e}"
                    )
                    batch_request.future.set_exception(e)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all requests in the batch
            for batch_request in batch_requests:
                try:
                    batch_request.future.set_exception(e)
                except Exception:
                    pass  # Future might already be done

    async def generate_stream(self, request: GenerationRequest):
        """
        Generate streaming response for a single request.

        Note: Streaming is not batched as it requires individual handling.

        Args:
            request: The generation request

        Yields:
            Streaming response chunks
        """
        if not self.is_running or self.engine is None:
            raise RuntimeError("Inference service not initialized")

        async for chunk in self.engine.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            repeat_penalty=request.repeat_penalty,
            seed=request.seed,
        ):
            yield chunk

    @property
    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests"""
        return self.is_running and self.engine is not None and self.engine.is_loaded


# Global inference service instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get the global inference service instance"""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
