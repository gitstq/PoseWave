"""
Signal Filter Module
Provides various filtering techniques for CSI signal preprocessing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    filter_type: str = "bandpass"
    order: int = 4
    lowcut: float = 0.5
    highcut: float = 10.0
    sample_rate: float = 100.0
    ripple_db: float = 0.5  # For Chebyshev filter


class SignalFilter:
    """
    Advanced Signal Filter for CSI Data.
    
    Provides multiple filtering strategies optimized for
    WiFi CSI signal preprocessing.
    
    Supported filter types:
        - bandpass: Butterworth bandpass filter
        - lowpass: Butterworth lowpass filter
        - highpass: Butterworth highpass filter
        - chebyshev: Chebyshev Type I filter
        - kalman: Kalman filter for noise reduction
        - wavelet: Wavelet denoising
    
    Example:
        >>> filt = SignalFilter()
        >>> clean_signal = filt.filter(noisy_signal, filter_type="bandpass")
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize Signal Filter.
        
        Args:
            config: Filter configuration parameters
        """
        self.config = config or FilterConfig()
        self._init_filters()
    
    def _init_filters(self) -> None:
        """Initialize filter coefficients."""
        nyquist = self.config.sample_rate / 2
        
        # Butterworth filter
        if self.config.filter_type == "bandpass":
            low = self.config.lowcut / nyquist
            high = self.config.highcut / nyquist
            self.b_butter, self.a_butter = signal.butter(
                self.config.order, [low, high], btype='band'
            )
        elif self.config.filter_type == "lowpass":
            cutoff = self.config.highcut / nyquist
            self.b_butter, self.a_butter = signal.butter(
                self.config.order, cutoff, btype='low'
            )
        elif self.config.filter_type == "highpass":
            cutoff = self.config.lowcut / nyquist
            self.b_butter, self.a_butter = signal.butter(
                self.config.order, cutoff, btype='high'
            )
        
        # Chebyshev filter
        if self.config.filter_type == "bandpass":
            low = self.config.lowcut / nyquist
            high = self.config.highcut / nyquist
            self.b_cheby, self.a_cheby = signal.cheby1(
                self.config.order,
                self.config.ripple_db,
                [low, high],
                btype='band'
            )
    
    def filter(
        self, 
        data: np.ndarray, 
        filter_type: str = "bandpass",
        method: str = "filtfilt"
    ) -> np.ndarray:
        """
        Apply filter to input signal.
        
        Args:
            data: Input signal array
            filter_type: Type of filter to apply
            method: Filtering method ('filtfilt' or 'lfilter')
        
        Returns:
            Filtered signal
        """
        if filter_type == "bandpass" or filter_type in ["lowpass", "highpass"]:
            b, a = self.b_butter, self.a_butter
        elif filter_type == "chebyshev":
            b, a = self.b_cheby, self.a_cheby
        elif filter_type == "kalman":
            return self._kalman_filter(data)
        elif filter_type == "wavelet":
            return self._wavelet_denoise(data)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        if method == "filtfilt":
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def _kalman_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply 1D Kalman filter for noise reduction.
        
        Simple implementation assuming constant signal with noise.
        """
        n = len(data)
        x = data[0]  # Initial estimate
        P = 1.0      # Initial error covariance
        Q = 0.001    # Process noise
        R = 0.1      # Measurement noise
        
        filtered = np.zeros(n)
        
        for i in range(n):
            # Prediction
            x_pred = x
            P_pred = P + Q
            
            # Update
            K = P_pred / (P_pred + R)
            x = x_pred + K * (data[i] - x_pred)
            P = (1 - K) * P_pred
            
            filtered[i] = x
        
        return filtered
    
    def _wavelet_denoise(
        self, 
        data: np.ndarray,
        wavelet: str = 'db4',
        level: int = 3
    ) -> np.ndarray:
        """
        Wavelet-based signal denoising.
        
        Uses soft thresholding with universal threshold.
        """
        try:
            import pywt
        except ImportError:
            logger.warning("PyWavelets not installed, falling back to Butterworth")
            return self.filter(data, "bandpass")
        
        # Decompose
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Universal threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Soft thresholding
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        
        # Reconstruct
        return pywt.waverec(coeffs, wavelet)[:len(data)]
    
    def median_filter(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter for spike removal."""
        return signal.medfilt(data, kernel_size=kernel_size)
    
    def moving_average(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply moving average smoothing."""
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')
    
    def savgol_smooth(
        self, 
        data: np.ndarray, 
        window: int = 11, 
        polyorder: int = 3
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter for smoothing."""
        return signal.savgol_filter(data, window, polyorder)
    
    def adaptive_filter(
        self, 
        data: np.ndarray, 
        noise_estimate: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Adaptive filtering based on local noise estimation.
        
        Automatically adjusts filter parameters based on
        local signal characteristics.
        """
        if noise_estimate is None:
            # Estimate noise from local variance
            window = min(20, len(data) // 10)
            local_var = np.array([
                np.var(data[max(0, i-window):min(len(data), i+window)])
                for i in range(len(data))
            ])
            noise_estimate = np.sqrt(local_var)
        
        # Adaptive threshold
        threshold = np.mean(noise_estimate) * 2
        
        # Apply stronger filtering where noise is high
        filtered = data.copy()
        high_noise = noise_estimate > threshold
        
        if np.any(high_noise):
            filtered[high_noise] = self._kalman_filter(data)[high_noise]
        
        return filtered
