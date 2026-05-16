"""
FastAPI Main Application
Main entry point for PoseWave API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/posewave', 1)[0])

from posewave.core.csi_processor import CSIProcessor, CSIConfig
from posewave.core.pose_detector import PoseDetector, DetectorConfig
from posewave.api.routes import router
from posewave.api.websocket import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
csi_processor: Optional[CSIProcessor] = None
pose_detector: Optional[PoseDetector] = None
ws_manager: Optional[ConnectionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global csi_processor, pose_detector, ws_manager
    
    # Startup
    logger.info("🚀 Starting PoseWave API Server...")
    
    csi_processor = CSIProcessor()
    pose_detector = PoseDetector()
    ws_manager = ConnectionManager()
    
    logger.info("✅ CSI Processor initialized")
    logger.info("✅ Pose Detector initialized")
    logger.info("✅ WebSocket Manager initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down PoseWave API Server...")


def create_app(
    title: str = "PoseWave API",
    version: str = "1.0.0",
    cors_origins: list = None
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        title: API title
        version: API version
        cors_origins: List of allowed CORS origins
    
    Returns:
        Configured FastAPI application
    """
    if cors_origins is None:
        cors_origins = ["*"]
    
    app = FastAPI(
        title=title,
        version=version,
        description="""
## 🌊 PoseWave - WiFi CSI Human Pose Detection API

Privacy-preserving human pose detection using WiFi Channel State Information.

### Features
- 🔒 **Privacy First**: No cameras required
- ⚡ **Real-time**: Millisecond-level detection
- 🏠 **Smart Home**: Fall detection, activity monitoring
- 📡 **WiFi-based**: Uses existing WiFi infrastructure

### Supported Poses
- Standing
- Sitting
- Lying
- Walking
- Falling (with alerts)
- Empty room detection
        """,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    return app


# Create default app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PoseWave API",
        "version": "1.0.0",
        "description": "WiFi CSI Human Pose Detection System",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "components": {
            "csi_processor": csi_processor is not None,
            "pose_detector": pose_detector is not None,
            "websocket": ws_manager is not None
        }
    }


def get_csi_processor() -> CSIProcessor:
    """Get CSI processor instance."""
    if csi_processor is None:
        raise RuntimeError("CSI Processor not initialized")
    return csi_processor


def get_pose_detector() -> PoseDetector:
    """Get Pose detector instance."""
    if pose_detector is None:
        raise RuntimeError("Pose Detector not initialized")
    return pose_detector


def get_ws_manager() -> ConnectionManager:
    """Get WebSocket manager instance."""
    if ws_manager is None:
        raise RuntimeError("WebSocket Manager not initialized")
    return ws_manager


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "posewave.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
