"""
API Routes
REST API endpoints for PoseWave.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import time
import logging

from posewave.api.main import get_csi_processor, get_pose_detector, get_ws_manager
from posewave.core.pose_detector import PoseType

logger = logging.getLogger(__name__)

router = APIRouter()


# ============== Request/Response Models ==============

class CSIDataRequest(BaseModel):
    """Request model for CSI data processing."""
    data: List[List[List[float]]] = Field(
        ...,
        description="CSI data as 3D array [subcarriers, antennas, samples]",
        example=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    sample_rate: Optional[float] = Field(
        default=100.0,
        description="Sample rate in Hz"
    )


class PoseResponse(BaseModel):
    """Response model for pose detection."""
    pose: str = Field(..., description="Detected pose type")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    timestamp: float = Field(..., description="Detection timestamp")
    velocity: Optional[float] = Field(None, description="Estimated velocity m/s")
    alerts: List[str] = Field(default_factory=list, description="Alert messages")


class FeaturesResponse(BaseModel):
    """Response model for feature extraction."""
    amplitude_features: Dict[str, Any] = Field(..., description="Amplitude features")
    phase_features: Dict[str, Any] = Field(..., description="Phase features")
    spectrogram_shape: List[int] = Field(..., description="Spectrogram dimensions")
    doppler_shape: List[int] = Field(..., description="Doppler dimensions")


class StatisticsResponse(BaseModel):
    """Response model for detection statistics."""
    total_detections: int
    pose_distribution: Dict[str, int]
    average_confidence: float
    current_pose: Optional[str]


class ConfigRequest(BaseModel):
    """Request model for configuration update."""
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    fall_threshold: Optional[float] = Field(None, gt=0.0)
    enable_alerts: Optional[bool] = None


# ============== Endpoints ==============

@router.post("/detect", response_model=PoseResponse)
async def detect_pose(
    request: CSIDataRequest,
    detector = Depends(get_pose_detector),
    processor = Depends(get_csi_processor)
):
    """
    Detect human pose from CSI data.
    
    Processes raw CSI data and returns detected pose with confidence score.
    
    - **data**: 3D array of CSI data [subcarriers, antennas, samples]
    - **sample_rate**: Sample rate in Hz (default: 100)
    """
    try:
        # Convert to numpy array
        csi_data = np.array(request.data)
        
        # Validate shape
        if csi_data.ndim not in [2, 3]:
            raise HTTPException(
                status_code=400,
                detail="CSI data must be 2D or 3D array"
            )
        
        # Process CSI data
        features = processor.process(csi_data)
        
        # Detect pose
        result = detector.detect(features)
        
        return PoseResponse(
            pose=result.pose_type.value,
            confidence=result.confidence,
            timestamp=result.timestamp,
            velocity=result.velocity,
            alerts=result.alerts
        )
        
    except Exception as e:
        logger.error(f"Error in pose detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features", response_model=FeaturesResponse)
async def extract_features(
    request: CSIDataRequest,
    processor = Depends(get_csi_processor)
):
    """
    Extract features from CSI data.
    
    Returns amplitude, phase, spectrogram, and Doppler features
    without performing pose detection.
    """
    try:
        csi_data = np.array(request.data)
        features = processor.process(csi_data)
        
        return FeaturesResponse(
            amplitude_features={
                'mean': float(np.mean(features['amplitude'])),
                'variance': float(np.var(features['amplitude'])),
                'shape': list(features['amplitude'].shape)
            },
            phase_features={
                'mean': float(np.mean(features['phase'])),
                'variance': float(np.var(features['phase'])),
                'shape': list(features['phase'].shape)
            },
            spectrogram_shape=list(features['spectrogram'].shape),
            doppler_shape=list(features['doppler'].shape)
        )
        
    except Exception as e:
        logger.error(f"Error in feature extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(detector = Depends(get_pose_detector)):
    """
    Get detection statistics.
    
    Returns statistics about recent detections including
    pose distribution and average confidence.
    """
    stats = detector.get_statistics()
    
    return StatisticsResponse(
        total_detections=stats.get('total_detections', 0),
        pose_distribution=stats.get('pose_distribution', {}),
        average_confidence=stats.get('average_confidence', 0.0),
        current_pose=stats.get('current_pose')
    )


@router.get("/poses")
async def list_poses():
    """
    List all supported pose types.
    
    Returns a list of all pose types that can be detected.
    """
    return {
        "poses": [
            {
                "name": pose.value,
                "description": _get_pose_description(pose)
            }
            for pose in PoseType
        ]
    }


@router.post("/config")
async def update_config(
    request: ConfigRequest,
    detector = Depends(get_pose_detector)
):
    """
    Update detector configuration.
    
    Allows runtime configuration of detection parameters.
    """
    try:
        if request.confidence_threshold is not None:
            detector.config.confidence_threshold = request.confidence_threshold
        
        if request.fall_threshold is not None:
            detector.config.fall_threshold = request.fall_threshold
        
        if request.enable_alerts is not None:
            detector.config.enable_alerts = request.enable_alerts
        
        return {
            "status": "updated",
            "config": {
                "confidence_threshold": detector.config.confidence_threshold,
                "fall_threshold": detector.config.fall_threshold,
                "enable_alerts": detector.config.enable_alerts
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_detector(
    detector = Depends(get_pose_detector),
    processor = Depends(get_csi_processor)
):
    """
    Reset detector and processor state.
    
    Clears detection history and data buffers.
    """
    try:
        detector.reset()
        processor.reset()
        
        return {"status": "reset", "message": "Detector and processor reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulate")
async def simulate_detection(
    pose: str = Query(
        default="standing",
        description="Pose type to simulate",
        enum=[p.value for p in PoseType]
    ),
    samples: int = Query(
        default=100,
        description="Number of samples to generate",
        ge=10,
        le=1000
    ),
    detector = Depends(get_pose_detector),
    processor = Depends(get_csi_processor)
):
    """
    Simulate pose detection with synthetic data.
    
    Useful for testing and demonstration without real CSI hardware.
    """
    try:
        # Generate synthetic CSI data based on pose
        csi_data = _generate_synthetic_csi(pose, samples)
        
        # Process and detect
        features = processor.process(csi_data)
        result = detector.detect(features)
        
        return {
            "input_pose": pose,
            "detected_pose": result.pose_type.value,
            "confidence": result.confidence,
            "samples": samples,
            "data_shape": list(csi_data.shape)
        }
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Helper Functions ==============

def _get_pose_description(pose: PoseType) -> str:
    """Get human-readable description for pose type."""
    descriptions = {
        PoseType.STANDING: "Person is standing upright",
        PoseType.SITTING: "Person is sitting down",
        PoseType.LYING: "Person is lying on the floor/bed",
        PoseType.WALKING: "Person is walking or moving",
        PoseType.FALLING: "Person is falling (alert triggered)",
        PoseType.EMPTY: "No person detected in the room",
        PoseType.UNKNOWN: "Unable to determine pose"
    }
    return descriptions.get(pose, "Unknown pose type")


def _generate_synthetic_csi(pose: str, samples: int) -> np.ndarray:
    """Generate synthetic CSI data for simulation."""
    np.random.seed(42)
    
    num_subcarriers = 64
    num_antennas = 3
    
    # Base signal
    base = np.random.randn(num_subcarriers, num_antennas, samples)
    
    # Add pose-specific characteristics
    if pose == "standing":
        # Moderate variance, some periodic component
        base += 0.3 * np.sin(np.linspace(0, 4*np.pi, samples))
        base *= 1.0
        
    elif pose == "sitting":
        # Lower variance
        base *= 0.7
        
    elif pose == "lying":
        # Very low variance
        base *= 0.3
        
    elif pose == "walking":
        # High variance, strong periodic component
        base += 0.8 * np.sin(np.linspace(0, 10*np.pi, samples))
        base *= 1.5
        
    elif pose == "falling":
        # Rapid change pattern
        base[:, :, :samples//2] *= 1.5
        base[:, :, samples//2:] *= 0.3
        
    elif pose == "empty":
        # Very low variance, almost no signal
        base *= 0.1
    
    return base.astype(np.float32)
