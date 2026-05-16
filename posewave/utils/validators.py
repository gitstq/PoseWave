"""
Validators Utility
Data validation functions for PoseWave.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import fields, is_dataclass
import logging

logger = logging.getLogger(__name__)


def validate_csi_data(
    data: np.ndarray,
    expected_subcarriers: int = 64,
    expected_antennas: int = 3,
    min_samples: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Validate CSI data array.
    
    Args:
        data: CSI data array to validate
        expected_subcarriers: Expected number of subcarriers
        expected_antennas: Expected number of antennas
        min_samples: Minimum required samples
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> csi_data = np.random.randn(64, 3, 100)
        >>> valid, error = validate_csi_data(csi_data)
    """
    if not isinstance(data, np.ndarray):
        return False, "Data must be a numpy array"
    
    if data.ndim not in [2, 3]:
        return False, f"Data must be 2D or 3D, got {data.ndim}D"
    
    # Check dimensions
    if data.ndim == 2:
        n_subcarriers, n_samples = data.shape
        n_antennas = 1
    else:
        n_subcarriers, n_antennas, n_samples = data.shape
    
    if n_subcarriers != expected_subcarriers:
        logger.warning(
            f"Unexpected subcarrier count: {n_subcarriers} "
            f"(expected {expected_subcarriers})"
        )
    
    if n_antennas != expected_antennas:
        logger.warning(
            f"Unexpected antenna count: {n_antennas} "
            f"(expected {expected_antennas})"
        )
    
    if n_samples < min_samples:
        return False, f"Insufficient samples: {n_samples} (minimum {min_samples})"
    
    # Check for NaN or Inf
    if np.any(np.isnan(data)):
        return False, "Data contains NaN values"
    
    if np.any(np.isinf(data)):
        return False, "Data contains Inf values"
    
    return True, None


def validate_config(
    config: Dict[str, Any],
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(config, dict):
        return False, "Config must be a dictionary"
    
    if required_keys:
        missing = [k for k in required_keys if k not in config]
        if missing:
            return False, f"Missing required keys: {missing}"
    
    return True, None


def validate_dataclass(
    obj: Any,
    dataclass_type: type
) -> Tuple[bool, Optional[str]]:
    """
    Validate that object matches dataclass structure.
    
    Args:
        obj: Object to validate
        dataclass_type: Expected dataclass type
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not is_dataclass(dataclass_type):
        return False, f"{dataclass_type} is not a dataclass"
    
    if not isinstance(obj, dataclass_type):
        return False, f"Expected {dataclass_type.__name__}, got {type(obj).__name__}"
    
    expected_fields = {f.name for f in fields(dataclass_type)}
    actual_fields = {f.name for f in fields(obj)}
    
    if expected_fields != actual_fields:
        return False, f"Field mismatch: expected {expected_fields}, got {actual_fields}"
    
    return True, None


def validate_pose_type(pose: str) -> bool:
    """
    Validate pose type string.
    
    Args:
        pose: Pose type string
    
    Returns:
        True if valid pose type
    """
    from posewave.core.pose_detector import PoseType
    
    valid_poses = [p.value for p in PoseType]
    return pose in valid_poses


def validate_sample_rate(
    sample_rate: float,
    min_rate: float = 1.0,
    max_rate: float = 1000.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate sample rate value.
    
    Args:
        sample_rate: Sample rate in Hz
        min_rate: Minimum allowed rate
        max_rate: Maximum allowed rate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(sample_rate, (int, float)):
        return False, "Sample rate must be a number"
    
    if sample_rate < min_rate:
        return False, f"Sample rate too low: {sample_rate} (minimum {min_rate})"
    
    if sample_rate > max_rate:
        return False, f"Sample rate too high: {sample_rate} (maximum {max_rate})"
    
    return True, None


def validate_threshold(
    threshold: float,
    name: str = "threshold",
    min_val: float = 0.0,
    max_val: float = 1.0
) -> Tuple[bool, Optional[str]]:
    """
    Validate threshold value.
    
    Args:
        threshold: Threshold value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(threshold, (int, float)):
        return False, f"{name} must be a number"
    
    if threshold < min_val or threshold > max_val:
        return False, f"{name} must be between {min_val} and {max_val}"
    
    return True, None


class DataValidator:
    """
    Comprehensive data validator for CSI data streams.
    
    Example:
        >>> validator = DataValidator()
        >>> is_valid = validator.validate(csi_data)
    """
    
    def __init__(
        self,
        expected_subcarriers: int = 64,
        expected_antennas: int = 3,
        min_samples: int = 10
    ):
        self.expected_subcarriers = expected_subcarriers
        self.expected_antennas = expected_antennas
        self.min_samples = min_samples
        self._validation_history: List[Dict[str, Any]] = []
    
    def validate(self, data: np.ndarray) -> bool:
        """
        Validate CSI data and log results.
        
        Args:
            data: CSI data to validate
        
        Returns:
            True if valid
        """
        is_valid, error = validate_csi_data(
            data,
            self.expected_subcarriers,
            self.expected_antennas,
            self.min_samples
        )
        
        self._validation_history.append({
            'is_valid': is_valid,
            'error': error,
            'shape': data.shape if isinstance(data, np.ndarray) else None
        })
        
        return is_valid
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self._validation_history:
            return {}
        
        total = len(self._validation_history)
        valid = sum(1 for v in self._validation_history if v['is_valid'])
        
        return {
            'total_validations': total,
            'valid_count': valid,
            'invalid_count': total - valid,
            'valid_rate': valid / total if total > 0 else 0
        }
    
    def reset(self) -> None:
        """Reset validation history."""
        self._validation_history.clear()
