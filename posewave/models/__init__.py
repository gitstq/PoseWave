"""
Models Module
Contains neural network models for pose detection.
"""

def __getattr__(name):
    if name == "WavePoseNet":
        from posewave.models.wavepose_net import WavePoseNet
        return WavePoseNet
    elif name == "FeatureExtractor":
        from posewave.models.feature_extractor import FeatureExtractor
        return FeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["WavePoseNet", "FeatureExtractor"]
