"""
Inference engine using llama-cpp-python optimized for Apple M2 GPU.

This module provides the core inference capabilities using llama.cpp
with Metal Performance Shaders (MPS) acceleration.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from llama_cpp import Llama, LlamaGrammar

from ..app.core.config import Settings

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine using llama-cpp-python.

    Optimized for Apple M2 GPU with Metal Performance Shaders (MPS).
    Supports both single and batch inference.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Optional[Llama] = None
        self.model_name = "unknown"
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the inference engine and load the model"""
        async with self._lock:
            if self.model is not None:
                logger.info("Model already loaded")
                return

            logger.info(f"Loading model from: {self.settings.model_path}")

            # Check if model file exists
            if not self.settings.model_exists:
                raise FileNotFoundError(
                    f"Model file not found: {self.settings.model_path}"
                )

            # Configure model parameters for Apple M2
            model_kwargs = {
                "model_path": self.settings.model_path,
                "n_gpu_layers": self.settings.n_gpu_layers,  # Offload to Metal GPU
                "n_ctx": self.settings.n_ctx,
                "n_batch": self.settings.n_batch,
                "use_mlock": self.settings.use_mlock,
                "use_mmap": self.settings.use_mmap,
                "verbose": self.settings.verbose,
                "seed": -1,  # Use random seed by default
            }

            # Add threads if specified
            if self.settings.n_threads:
                model_kwargs["n_threads"] = self.settings.n_threads

            try:
                # Load model in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, lambda: Llama(**model_kwargs)
                )

                # Extract model name from path
                self.model_name = self.settings.model_path.split("/")[-1]

                logger.info(f"Model loaded successfully: {self.model_name}")
                logger.info(f"Context size: {self.settings.n_ctx}")
                logger.info(f"GPU layers: {self.settings.n_gpu_layers}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    async def shutdown(self):
        """Shutdown the inference engine"""
        async with self._lock:
            if self.model is not None:
                # llama-cpp-python handles cleanup automatically
                self.model = None
                logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            top_k: Top-k sampling
            stop: Stop sequences
            repeat_penalty: Repetition penalty
            seed: Random seed for reproducibility

        Returns:
            List of generation results
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if not prompts:
            return []

        logger.debug(f"Generating batch of {len(prompts)} prompts")

        # Process prompts sequentially within the batch call
        # This resolves issues with the underlying library's batch handling
        # while still benefiting from the service-level dynamic batching.
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = await self._generate_single(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error generating for prompt {i}: {e}")
                # Return error result for this prompt
                results.append(
                    {
                        "id": str(uuid.uuid4()),
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [
                            {
                                "text": f"Error: {str(e)}",
                                "finish_reason": "error",
                                "index": i,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    }
                )

        return results

    async def _generate_single(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text for a single prompt"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        generation_id = str(uuid.uuid4())
        created_time = int(time.time())

        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": False,
        }

        if stop:
            generation_kwargs["stop"] = stop

        if seed is not None:
            generation_kwargs["seed"] = seed

        try:
            # Generate in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.model(**generation_kwargs)
            )

            # Format response
            choice = result["choices"][0]

            return {
                "id": generation_id,
                "object": "text_completion",
                "created": created_time,
                "model": self.model_name,
                "choices": [
                    {
                        "text": choice["text"],
                        "finish_reason": choice["finish_reason"],
                        "index": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"],
                    "total_tokens": result["usage"]["total_tokens"],
                },
            }

        except Exception as e:
            logger.error(f"Error in single generation: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming text for a single prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p nucleus sampling
            top_k: Top-k sampling
            stop: Stop sequences
            repeat_penalty: Repetition penalty
            seed: Random seed for reproducibility

        Yields:
            Streaming response chunks
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        generation_id = str(uuid.uuid4())
        created_time = int(time.time())

        # Prepare generation parameters
        generation_kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": True,
        }

        if stop:
            generation_kwargs["stop"] = stop

        if seed is not None:
            generation_kwargs["seed"] = seed

        try:
            # Create streaming generator in thread pool
            loop = asyncio.get_event_loop()

            # We need to handle streaming in a more complex way
            # since we can't directly await a generator
            stream_queue = asyncio.Queue()

            async def stream_worker():
                """Worker function to handle streaming in thread pool"""
                try:

                    def run_stream():
                        return self.model(**generation_kwargs)

                    stream_iter = await loop.run_in_executor(None, run_stream)

                    for chunk in stream_iter:
                        await stream_queue.put(chunk)

                    await stream_queue.put(None)  # Signal end

                except Exception as e:
                    await stream_queue.put(e)

            # Start the worker
            worker_task = asyncio.create_task(stream_worker())

            try:
                while True:
                    chunk = await stream_queue.get()

                    if chunk is None:  # End of stream
                        break

                    if isinstance(chunk, Exception):
                        raise chunk

                    # Format streaming chunk
                    choice = chunk["choices"][0]

                    yield {
                        "id": generation_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": self.model_name,
                        "choices": [
                            {
                                "text": choice["text"],
                                "finish_reason": choice.get("finish_reason"),
                                "index": 0,
                            }
                        ],
                    }

            finally:
                # Clean up worker task
                if not worker_task.done():
                    worker_task.cancel()
                    try:
                        await worker_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_path": self.settings.model_path,
            "model_name": self.model_name,
            "n_ctx": self.settings.n_ctx,
            "n_gpu_layers": self.settings.n_gpu_layers,
            "n_batch": self.settings.n_batch,
        }
