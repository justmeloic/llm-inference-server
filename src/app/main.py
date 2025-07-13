"""
Main FastAPI application for the LLM inference server.

This module initializes the FastAPI application, configures middleware,
and sets up the inference service with proper lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from ..lib.core.banner import print_banner
from ..lib.core.config import get_settings
from .api.v1.routes import router as v1_router
from .services.inference_service import get_inference_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)
console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles startup and shutdown of the inference service.
    """
    console.clear()
    with console.status("[bold green]Starting server...", spinner="dots") as status:
        try:
            # Load settings
            status.update("[bold green]Loading configuration...")
            settings = get_settings()
            logger.info(f"Loading configuration from: {settings.model_path}")

            # Initialize service
            status.update("[bold green]Initializing inference service...")
            service = get_inference_service()

            # Load the model
            model_name = settings.model_path.split("/")[-1]
            status.update(
                f"[bold green]Loading model ([cyan]{model_name}[/cyan])... (This may take a moment)"
            )
            await asyncio.sleep(
                0.1
            )  # Ensures the status message renders before blocking
            await service.initialize()

            # Clear the console and print the banner
            console.clear()
            print_banner(console, settings)

            logger.info("✅ LLM Inference Server is ready to accept requests.")

        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            raise

        yield

        # Shutdown
        try:
            logger.info("Shutting down LLM Inference Server...")
            service = get_inference_service()
            await service.shutdown()
            logger.info("✅ Server shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    settings = get_settings()

    app = FastAPI(
        title="LLM Inference Server",
        description="High-performance LLM inference server optimized for Apple M2 with dynamic batching",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(v1_router, prefix="/api/v1", tags=["Generation"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc)
                if settings.verbose
                else "An unexpected error occurred",
            },
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with server information"""
        return {
            "message": "LLM Inference Server",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "api": "/api/v1",
        }

    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    # Run the server
    uvicorn.run(
        "src.app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,  # Set to True for development
        access_log=True,
        log_level="info",
    )
