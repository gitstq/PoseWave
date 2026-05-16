"""
PoseWave - WiFi CSI Human Pose Detection System
A privacy-preserving human pose detection system using WiFi Channel State Information.

Author: PoseWave Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "PoseWave Team"

from posewave.core.csi_processor import CSIProcessor
from posewave.core.pose_detector import PoseDetector
from posewave.models.wavepose_net import WavePoseNet

__all__ = [
    "CSIProcessor",
    "PoseDetector", 
    "WavePoseNet",
    "__version__",
]
