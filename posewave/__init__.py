"""
PoseWave - WiFi CSI Human Pose Detection System
A privacy-preserving human pose detection system using WiFi Channel State Information.

Author: PoseWave Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "PoseWave Team"

# Lazy imports to avoid torch dependency at import time
def __getattr__(name):
    if name == "CSIProcessor":
        from posewave.core.csi_processor import CSIProcessor
        return CSIProcessor
    elif name == "PoseDetector":
        from posewave.core.pose_detector import PoseDetector
        return PoseDetector
    elif name == "WavePoseNet":
        from posewave.models.wavepose_net import WavePoseNet
        return WavePoseNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CSIProcessor",
    "PoseDetector", 
    "WavePoseNet",
    "__version__",
]
