"""
Feature Extractor Module
Extracts handcrafted features from CSI data for classical ML approaches.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    window_sizes: Tuple[int, ...] = (10, 30, 50)
    statistical_features: bool = True
    spectral_features: bool = True
    temporal_features: bool = True


class FeatureExtractor:
    """
    Handcrafted Feature Extractor for WiFi CSI Data.
    
    Extracts various features from CSI signals for pose detection:
        - Statistical features (mean, variance, skewness, kurtosis)
        - Spectral features (FFT components, spectral entropy)
        - Temporal features (autocorrelation, zero crossing rate)
    
    Can be used with classical ML models (SVM, Random Forest) or
    as additional features for deep learning models.
    
    Example:
        >>> extractor = FeatureExtractor()
        >>> csi_data = np.random.randn(64, 3, 100)
        >>> features = extractor.extract(csi_data)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize Feature Extractor.
        
        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
        self._feature_names: List[str] = []
    
    def extract(self, csi_data: np.ndarray) -> np.ndarray:
        """
        Extract all features from CSI data.
        
        Args:
            csi_data: CSI data array of shape (subcarriers, antennas, samples)
        
        Returns:
            Feature vector
        """
        features = []
        self._feature_names = []
        
        # Ensure 3D shape
        if csi_data.ndim == 2:
            csi_data = csi_data[:, np.newaxis, :]
        
        # Amplitude features
        amplitude = np.abs(csi_data)
        
        if self.config.statistical_features:
            stat_features = self._extract_statistical(amplitude)
            features.extend(stat_features)
        
        if self.config.spectral_features:
            spec_features = self._extract_spectral(amplitude)
            features.extend(spec_features)
        
        if self.config.temporal_features:
            temp_features = self._extract_temporal(amplitude)
            features.extend(temp_features)
        
        return np.array(features)
    
    def extract_with_names(
        self, 
        csi_data: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features with corresponding names.
        
        Args:
            csi_data: CSI data array
        
        Returns:
            Tuple of (feature_vector, feature_names)
        """
        features = self.extract(csi_data)
        return features, self._feature_names.copy()
    
    def _extract_statistical(self, amplitude: np.ndarray) -> List[float]:
        """Extract statistical features."""
        features = []
        
        # Global statistics
        features.extend([
            np.mean(amplitude),
            np.std(amplitude),
            np.var(amplitude),
            stats.skew(amplitude.flatten()),
            stats.kurtosis(amplitude.flatten()),
            np.max(amplitude),
            np.min(amplitude),
            np.median(amplitude),
        ])
        self._feature_names.extend([
            'mean', 'std', 'var', 'skewness', 'kurtosis',
            'max', 'min', 'median'
        ])
        
        # Per-antenna statistics
        for ant in range(amplitude.shape[1]):
            ant_data = amplitude[:, ant, :].flatten()
            features.extend([
                np.mean(ant_data),
                np.std(ant_data),
                np.var(ant_data),
            ])
            self._feature_names.extend([
                f'ant{ant}_mean', f'ant{ant}_std', f'ant{ant}_var'
            ])
        
        # Per-subcarrier variance
        sc_var = np.var(amplitude, axis=-1).flatten()
        features.extend([
            np.mean(sc_var),
            np.std(sc_var),
            np.max(sc_var),
        ])
        self._feature_names.extend([
            'sc_var_mean', 'sc_var_std', 'sc_var_max'
        ])
        
        return features
    
    def _extract_spectral(self, amplitude: np.ndarray) -> List[float]:
        """Extract spectral features using FFT."""
        features = []
        
        # Average across subcarriers and antennas
        avg_signal = amplitude.mean(axis=(0, 1))
        
        # FFT
        fft_vals = np.abs(np.fft.rfft(avg_signal))
        fft_freqs = np.fft.rfftfreq(len(avg_signal))
        
        # Spectral centroid
        spectral_centroid = np.sum(fft_freqs * fft_vals) / np.sum(fft_vals)
        features.append(spectral_centroid)
        self._feature_names.append('spectral_centroid')
        
        # Spectral bandwidth
        spectral_bw = np.sqrt(
            np.sum(((fft_freqs - spectral_centroid) ** 2) * fft_vals) / np.sum(fft_vals)
        )
        features.append(spectral_bw)
        self._feature_names.append('spectral_bandwidth')
        
        # Spectral entropy
        psd = fft_vals / np.sum(fft_vals)
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-10))
        features.append(spectral_entropy)
        self._feature_names.append('spectral_entropy')
        
        # Peak frequencies
        peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals) * 0.1)
        if len(peaks) > 0:
            peak_freqs = fft_freqs[peaks]
            features.extend([
                np.mean(peak_freqs),
                np.max(peak_freqs),
                len(peaks),
            ])
        else:
            features.extend([0.0, 0.0, 0])
        self._feature_names.extend([
            'peak_freq_mean', 'peak_freq_max', 'num_peaks'
        ])
        
        # Energy in frequency bands
        low_band = (fft_freqs >= 0.5) & (fft_freqs < 2)
        mid_band = (fft_freqs >= 2) & (fft_freqs < 5)
        high_band = (fft_freqs >= 5) & (fft_freqs < 10)
        
        features.extend([
            np.sum(fft_vals[low_band] ** 2),
            np.sum(fft_vals[mid_band] ** 2),
            np.sum(fft_vals[high_band] ** 2),
        ])
        self._feature_names.extend([
            'low_band_energy', 'mid_band_energy', 'high_band_energy'
        ])
        
        return features
    
    def _extract_temporal(self, amplitude: np.ndarray) -> List[float]:
        """Extract temporal features."""
        features = []
        
        # Average signal
        avg_signal = amplitude.mean(axis=(0, 1))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(avg_signal)) != 0)
        zcr = zero_crossings / len(avg_signal)
        features.append(zcr)
        self._feature_names.append('zero_crossing_rate')
        
        # Autocorrelation at different lags
        for lag in [1, 5, 10, 20]:
            if lag < len(avg_signal):
                autocorr = np.corrcoef(avg_signal[:-lag], avg_signal[lag:])[0, 1]
                features.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                features.append(0)
            self._feature_names.append(f'autocorr_lag{lag}')
        
        # Signal derivative features
        derivative = np.diff(avg_signal)
        features.extend([
            np.mean(np.abs(derivative)),
            np.std(derivative),
            np.max(np.abs(derivative)),
        ])
        self._feature_names.extend([
            'derivative_mean', 'derivative_std', 'derivative_max'
        ])
        
        # Rolling statistics
        for window in self.config.window_sizes:
            if window < len(avg_signal):
                rolling_mean = np.convolve(
                    avg_signal, 
                    np.ones(window)/window, 
                    mode='valid'
                )
                features.extend([
                    np.std(rolling_mean),
                    np.max(rolling_mean) - np.min(rolling_mean),
                ])
            else:
                features.extend([0, 0])
            self._feature_names.extend([
                f'rolling_std_w{window}', f'rolling_range_w{window}'
            ])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names from last extraction."""
        return self._feature_names.copy()
    
    def get_feature_count(self) -> int:
        """Get total number of features extracted."""
        return len(self._feature_names)
