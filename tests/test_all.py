"""
Tests Module
Unit tests for PoseWave components.
"""

import pytest
import numpy as np
from typing import Dict, Any

# Import PoseWave components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from posewave.core.csi_processor import CSIProcessor, CSIConfig
from posewave.core.pose_detector import PoseDetector, DetectorConfig, PoseType
from posewave.core.signal_filter import SignalFilter, FilterConfig
from posewave.models.wavepose_net import WavePoseNet, ModelConfig
from posewave.models.feature_extractor import FeatureExtractor, FeatureConfig


class TestCSIProcessor:
    """Tests for CSI Processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CSIConfig()
        self.processor = CSIProcessor(self.config)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.config.num_subcarriers == 64
        assert self.processor.config.num_antennas == 3
        assert self.processor.buffer.shape == (64, 3, 100)
    
    def test_process_2d_data(self):
        """Test processing 2D CSI data."""
        csi_data = np.random.randn(64, 100)
        features = self.processor.process(csi_data)
        
        assert 'amplitude' in features
        assert 'phase' in features
        assert 'spectrogram' in features
        assert 'doppler' in features
    
    def test_process_3d_data(self):
        """Test processing 3D CSI data."""
        csi_data = np.random.randn(64, 3, 100)
        features = self.processor.process(csi_data)
        
        assert features['amplitude'].shape[-1] == 5  # 5 amplitude features
        assert features['phase'].shape[-1] == 2  # 2 phase features
    
    def test_reset(self):
        """Test processor reset."""
        csi_data = np.random.randn(64, 3, 100)
        self.processor.process(csi_data)
        
        self.processor.reset()
        
        assert np.allclose(self.processor.buffer, 0)


class TestPoseDetector:
    """Tests for Pose Detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DetectorConfig()
        self.detector = PoseDetector(self.config)
    
    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.config.confidence_threshold == 0.7
        assert len(self.detector.history) == 0
    
    def test_detect_standing(self):
        """Test standing pose detection."""
        features = {
            'amplitude': np.random.randn(64, 3, 5) * 0.3 + 0.5,
            'phase': np.random.randn(64, 3, 2) * 0.1,
            'spectrogram': np.random.randn(17, 9),
            'doppler': np.random.randn(3, 50) * 0.05
        }
        
        result = self.detector.detect(features)
        
        assert result.pose_type in PoseType
        assert 0 <= result.confidence <= 1
        assert result.timestamp > 0
    
    def test_detect_walking(self):
        """Test walking pose detection."""
        features = {
            'amplitude': np.random.randn(64, 3, 5) * 0.8 + 1.0,
            'phase': np.random.randn(64, 3, 2) * 0.3,
            'spectrogram': np.random.randn(17, 9) * 0.5,
            'doppler': np.random.randn(3, 50) * 0.8
        }
        
        result = self.detector.detect(features)
        
        assert result.pose_type in PoseType
        assert result.velocity is not None
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        features = {
            'amplitude': np.random.randn(64, 3, 5),
            'phase': np.random.randn(64, 3, 2),
            'spectrogram': np.random.randn(17, 9),
            'doppler': np.random.randn(3, 50)
        }
        
        for _ in range(5):
            self.detector.detect(features)
        
        stats = self.detector.get_statistics()
        
        assert stats['total_detections'] == 5
        assert 'pose_distribution' in stats
    
    def test_reset(self):
        """Test detector reset."""
        features = {
            'amplitude': np.random.randn(64, 3, 5),
            'phase': np.random.randn(64, 3, 2),
            'spectrogram': np.random.randn(17, 9),
            'doppler': np.random.randn(3, 50)
        }
        
        self.detector.detect(features)
        self.detector.reset()
        
        assert len(self.detector.history) == 0


class TestSignalFilter:
    """Tests for Signal Filter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FilterConfig()
        self.filter = SignalFilter(self.config)
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        signal = np.random.randn(100)
        filtered = self.filter.filter(signal, filter_type="bandpass")
        
        assert filtered.shape == signal.shape
        assert not np.allclose(filtered, signal)
    
    def test_kalman_filter(self):
        """Test Kalman filtering."""
        signal = np.random.randn(100)
        filtered = self.filter.filter(signal, filter_type="kalman")
        
        assert filtered.shape == signal.shape
    
    def test_moving_average(self):
        """Test moving average smoothing."""
        signal = np.random.randn(100)
        smoothed = self.filter.moving_average(signal, window=5)
        
        assert smoothed.shape == signal.shape


class TestWavePoseNet:
    """Tests for WavePoseNet model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ModelConfig()
        self.model = WavePoseNet(self.config)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.config.num_classes == 6
        assert self.model.config.hidden_size == 128
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        x = torch.randn(batch_size, 64, 3, 100)
        
        output = self.model(x)
        
        assert output.shape == (batch_size, self.config.num_classes)
    
    def test_predict(self):
        """Test prediction method."""
        x = torch.randn(1, 64, 3, 100)
        
        predictions, confidence = self.model.predict(x)
        
        assert predictions.shape == (1,)
        assert confidence.shape == (1,)
        assert 0 <= confidence[0] <= 1
    
    def test_parameter_count(self):
        """Test parameter counting."""
        count = self.model.count_parameters()
        
        assert count > 0


class TestFeatureExtractor:
    """Tests for Feature Extractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = FeatureConfig()
        self.extractor = FeatureExtractor(self.config)
    
    def test_extract_features(self):
        """Test feature extraction."""
        csi_data = np.random.randn(64, 3, 100)
        features = self.extractor.extract(csi_data)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_extract_with_names(self):
        """Test feature extraction with names."""
        csi_data = np.random.randn(64, 3, 100)
        features, names = self.extractor.extract_with_names(csi_data)
        
        assert len(features) == len(names)
        assert 'mean' in names
        assert 'spectral_centroid' in names


# Import torch for model tests
import torch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
