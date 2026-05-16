"""
Models Module
Contains neural network models for pose detection.
"""

from posewave.models.wavepose_net import WavePoseNet
from posewave.models.feature_extractor import FeatureExtractor

__all__ = ["WavePoseNet", "FeatureExtractor"]
