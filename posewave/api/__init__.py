"""
API Module
FastAPI-based REST API for PoseWave.
"""

from posewave.api.main import app, create_app
from posewave.api.routes import router
from posewave.api.websocket import ConnectionManager

__all__ = ["app", "create_app", "router", "ConnectionManager"]
