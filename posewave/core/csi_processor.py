"""
CSI (Channel State Information) Processor
Handles WiFi CSI data acquisition, preprocessing, and feature extraction.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, fftfreq
import logging

logger = logging.getLogger(__name__)


@dataclass
class CSIConfig:
    """Configuration for CSI processing."""
    num_subcarriers: int = 64
    num_antennas: int = 3
    sample_rate: float = 100.0  # Hz
    window_size: int = 100
    overlap: float = 0.5
    lowcut: float = 0.5  # Hz
    highcut: float = 10.0  # Hz


class CSIProcessor:
    """
    WiFi CSI Data Processor.
    
    Processes raw CSI data from WiFi signals for human pose detection.
    Supports multiple antenna configurations and subcarrier selections.
    
    Attributes:
        config: CSI processing configuration
        buffer: Rolling buffer for CSI data
        filter_coeffs: Bandpass filter coefficients
    
    Example:
        >>> processor = CSIProcessor()
        >>> csi_data = np.random.randn(64, 3, 100)  # 64 subcarriers, 3 antennas, 100 samples
        >>> features = processor.process(csi_data)
    """
    
    def __init__(self, config: Optional[CSIConfig] = None):
        """
        Initialize CSI Processor.
        
        Args:
            config: Optional CSI configuration. Uses defaults if not provided.
        """
        self.config = config or CSIConfig()
        self.buffer = np.zeros((
            self.config.num_subcarriers,
            self.config.num_antennas,
            self.config.window_size
        ))
        self._init_filter()
        logger.info(f"CSIProcessor initialized with {self.config.num_subcarriers} subcarriers")
    
    def _init_filter(self) -> None:
        """Initialize bandpass filter for signal denoising."""
        nyquist = self.config.sample_rate / 2
        low = self.config.lowcut / nyquist
        high = self.config.highcut / nyquist
        self.filter_b, self.filter_a = signal.butter(
            4, [low, high], btype='band'
        )
    
    def process(self, csi_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process raw CSI data and extract features.
        
        Args:
            csi_data: Raw CSI data with shape (subcarriers, antennas, samples)
                     or (subcarriers, samples) for single antenna.
        
        Returns:
            Dictionary containing extracted features:
                - amplitude: Amplitude features
                - phase: Phase features (cleaned)
                - spectrogram: Time-frequency representation
                - doppler: Doppler shift estimates
        """
        # Ensure 3D shape
        if csi_data.ndim == 2:
            csi_data = csi_data[:, np.newaxis, :]
        
        # Update buffer
        self._update_buffer(csi_data)
        
        # Extract amplitude and phase
        amplitude = np.abs(self.buffer)
        phase = self._extract_phase(self.buffer)
        
        # Apply bandpass filter
        filtered_amp = self._apply_filter(amplitude)
        
        # Extract features
        features = {
            'amplitude': self._extract_amplitude_features(filtered_amp),
            'phase': self._extract_phase_features(phase),
            'spectrogram': self._compute_spectrogram(filtered_amp),
            'doppler': self._estimate_doppler(filtered_amp),
        }
        
        return features
    
    def _update_buffer(self, csi_data: np.ndarray) -> None:
        """Update rolling buffer with new CSI data."""
        n_samples = csi_data.shape[-1]
        if n_samples >= self.config.window_size:
            self.buffer = csi_data[..., -self.config.window_size:]
        else:
            self.buffer = np.roll(self.buffer, -n_samples, axis=-1)
            self.buffer[..., -n_samples:] = csi_data
    
    def _extract_phase(self, csi_data: np.ndarray) -> np.ndarray:
        """
        Extract and clean phase from CSI data.
        
        Applies phase unwrapping and linear fitting to remove
        phase offsets caused by timing errors.
        """
        raw_phase = np.angle(csi_data)
        
        # Unwrap phase
        unwrapped_phase = np.unwrap(raw_phase, axis=0)
        
        # Linear fitting to remove offsets
        for ant in range(unwrapped_phase.shape[1]):
            for t in range(unwrapped_phase.shape[2]):
                x = np.arange(unwrapped_phase.shape[0])
                y = unwrapped_phase[:, ant, t]
                coeffs = np.polyfit(x, y, 1)
                unwrapped_phase[:, ant, t] = y - np.polyval(coeffs, x)
        
        return unwrapped_phase
    
    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to CSI amplitude."""
        filtered = np.zeros_like(data)
        for sc in range(data.shape[0]):
            for ant in range(data.shape[1]):
                filtered[sc, ant, :] = signal.filtfilt(
                    self.filter_b, self.filter_a, data[sc, ant, :]
                )
        return filtered
    
    def _extract_amplitude_features(self, amplitude: np.ndarray) -> np.ndarray:
        """Extract statistical features from amplitude data."""
        features = []
        
        # Mean and variance across time
        features.append(np.mean(amplitude, axis=-1))
        features.append(np.var(amplitude, axis=-1))
        
        # Max and min
        features.append(np.max(amplitude, axis=-1))
        features.append(np.min(amplitude, axis=-1))
        
        # Energy
        features.append(np.sum(amplitude ** 2, axis=-1))
        
        return np.stack(features, axis=-1)
    
    def _extract_phase_features(self, phase: np.ndarray) -> np.ndarray:
        """Extract features from cleaned phase data."""
        features = []
        
        # Phase variance
        features.append(np.var(phase, axis=-1))
        
        # Phase change rate
        phase_diff = np.diff(phase, axis=-1)
        features.append(np.mean(np.abs(phase_diff), axis=-1))
        
        return np.stack(features, axis=-1)
    
    def _compute_spectrogram(self, amplitude: np.ndarray) -> np.ndarray:
        """Compute time-frequency spectrogram."""
        # Use first antenna for spectrogram
        sig = amplitude[:, 0, :].mean(axis=0)  # Average across subcarriers
        
        f, t, Sxx = signal.spectrogram(
            sig, 
            fs=self.config.sample_rate,
            nperseg=32,
            noverlap=16
        )
        
        return Sxx
    
    def _estimate_doppler(self, amplitude: np.ndarray) -> np.ndarray:
        """Estimate Doppler shift from amplitude variations."""
        # Average amplitude across subcarriers
        avg_amp = amplitude.mean(axis=0)
        
        # Compute FFT
        n = avg_amp.shape[-1]
        fft_vals = fft(avg_amp, axis=-1)
        freqs = fftfreq(n, 1/self.config.sample_rate)
        
        # Get positive frequencies
        pos_mask = freqs > 0
        doppler = np.abs(fft_vals[:, pos_mask])
        
        return doppler
    
    def reset(self) -> None:
        """Reset the processor state."""
        self.buffer = np.zeros((
            self.config.num_subcarriers,
            self.config.num_antennas,
            self.config.window_size
        ))
        logger.info("CSIProcessor reset")
