"""
Pose Detector Module
Detects human poses from processed CSI features.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class PoseType(Enum):
    """Supported pose types for detection."""
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    WALKING = "walking"
    FALLING = "falling"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class PoseResult:
    """Result of pose detection."""
    pose_type: PoseType
    confidence: float
    timestamp: float
    keypoints: Optional[Dict[str, Tuple[float, float]]] = None
    velocity: Optional[float] = None
    duration: Optional[float] = None
    alerts: List[str] = field(default_factory=list)


@dataclass
class DetectorConfig:
    """Configuration for pose detector."""
    confidence_threshold: float = 0.7
    fall_threshold: float = 2.0  # m/s²
    stillness_threshold: float = 0.1
    min_detection_interval: float = 0.1  # seconds
    history_size: int = 30
    enable_alerts: bool = True


class PoseDetector:
    """
    Human Pose Detector using WiFi CSI Features.
    
    Detects various human poses and activities from processed
    CSI signal features without requiring cameras.
    
    Features:
        - Real-time pose classification
        - Fall detection with alerts
        - Activity tracking
        - Privacy-preserving (no video)
    
    Example:
        >>> detector = PoseDetector()
        >>> features = processor.process(csi_data)
        >>> result = detector.detect(features)
        >>> print(result.pose_type, result.confidence)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize Pose Detector.
        
        Args:
            config: Detector configuration parameters
        """
        self.config = config or DetectorConfig()
        self.history: List[PoseResult] = []
        self._last_detection_time = 0.0
        self._pose_start_time: Optional[float] = None
        self._current_pose: Optional[PoseType] = None
        
        # Feature thresholds (learned from calibration)
        self._thresholds = {
            'amplitude_variance_standing': (0.1, 0.5),
            'amplitude_variance_sitting': (0.05, 0.2),
            'amplitude_variance_walking': (0.5, 2.0),
            'doppler_standing': (0.0, 0.1),
            'doppler_walking': (0.3, 1.5),
            'spectrogram_energy_still': (0.0, 0.05),
            'spectrogram_energy_moving': (0.1, 1.0),
        }
        
        logger.info("PoseDetector initialized")
    
    def detect(self, features: Dict[str, np.ndarray]) -> PoseResult:
        """
        Detect human pose from CSI features.
        
        Args:
            features: Dictionary of extracted CSI features
        
        Returns:
            PoseResult containing detected pose and metadata
        """
        current_time = time.time()
        
        # Rate limiting
        if current_time - self._last_detection_time < self.config.min_detection_interval:
            return self.history[-1] if self.history else self._unknown_result()
        
        self._last_detection_time = current_time
        
        # Extract key features
        amp_features = features.get('amplitude', np.zeros((64, 3, 5)))
        phase_features = features.get('phase', np.zeros((64, 3, 2)))
        spectrogram = features.get('spectrogram', np.zeros((17, 9)))
        doppler = features.get('doppler', np.zeros((3, 50)))
        
        # Compute aggregate metrics
        amp_var = np.mean(np.var(amp_features, axis=-1))
        amp_energy = np.mean(amp_features ** 2)
        doppler_mean = np.mean(np.abs(doppler)) if doppler.size > 0 else 0
        spec_energy = np.mean(spectrogram ** 2) if spectrogram.size > 0 else 0
        
        # Classify pose
        pose_type, confidence = self._classify_pose(
            amp_var, amp_energy, doppler_mean, spec_energy
        )
        
        # Detect fall
        alerts = []
        if self.config.enable_alerts:
            if pose_type == PoseType.FALLING:
                alerts.append("⚠️ FALL DETECTED!")
            elif pose_type == PoseType.LYING and self._check_fall_pattern():
                alerts.append("⚠️ Possible fall - person lying down suddenly")
        
        # Compute velocity estimate
        velocity = self._estimate_velocity(doppler_mean)
        
        # Create result
        result = PoseResult(
            pose_type=pose_type,
            confidence=confidence,
            timestamp=current_time,
            velocity=velocity,
            alerts=alerts
        )
        
        # Update history
        self._update_history(result)
        
        # Track pose duration
        self._track_pose_duration(pose_type, current_time)
        
        return result
    
    def _classify_pose(
        self,
        amp_var: float,
        amp_energy: float,
        doppler: float,
        spec_energy: float
    ) -> Tuple[PoseType, float]:
        """
        Classify pose based on feature metrics.
        
        Uses rule-based classification with confidence scoring.
        """
        scores = {}
        
        # Empty room detection
        if amp_var < 0.01 and doppler < 0.01:
            scores[PoseType.EMPTY] = 0.9
        
        # Walking detection
        if doppler > 0.3 and amp_var > 0.5:
            scores[PoseType.WALKING] = min(1.0, doppler / 1.5)
        
        # Standing detection
        if 0.1 < amp_var < 0.5 and doppler < 0.2:
            scores[PoseType.STANDING] = 0.8
        
        # Sitting detection
        if 0.05 < amp_var < 0.2 and doppler < 0.1:
            scores[PoseType.SITTING] = 0.75
        
        # Lying detection
        if amp_var < 0.05 and doppler < 0.05 and amp_energy < 0.5:
            scores[PoseType.LYING] = 0.7
        
        # Falling detection (rapid change)
        if self._detect_fall(amp_var, doppler):
            scores[PoseType.FALLING] = 0.85
        
        if not scores:
            return PoseType.UNKNOWN, 0.5
        
        # Get best match
        best_pose = max(scores, key=scores.get)
        confidence = scores[best_pose]
        
        return best_pose, confidence
    
    def _detect_fall(self, amp_var: float, doppler: float) -> bool:
        """Detect fall from rapid signal changes."""
        if len(self.history) < 3:
            return False
        
        # Check for rapid transition from standing/walking to low activity
        recent_poses = [h.pose_type for h in self.history[-5:]]
        
        if PoseType.STANDING in recent_poses or PoseType.WALKING in recent_poses:
            # Check for sudden drop in activity
            recent_doppler = [h.velocity or 0 for h in self.history[-3:]]
            if len(recent_doppler) >= 2:
                if recent_doppler[-1] < 0.1 and max(recent_doppler[:-1]) > 0.3:
                    return True
        
        return False
    
    def _check_fall_pattern(self) -> bool:
        """Check if lying pose follows a fall pattern."""
        if len(self.history) < 5:
            return False
        
        recent = self.history[-5:]
        # Fall pattern: standing/walking -> sudden drop -> lying
        for i, r in enumerate(recent[:-1]):
            if r.pose_type in [PoseType.STANDING, PoseType.WALKING]:
                if recent[i+1].pose_type == PoseType.LYING:
                    return True
        
        return False
    
    def _estimate_velocity(self, doppler: float) -> float:
        """Estimate movement velocity from Doppler shift."""
        # Simplified velocity estimation
        # Real implementation would use proper Doppler-to-velocity conversion
        return doppler * 0.5  # m/s approximation
    
    def _update_history(self, result: PoseResult) -> None:
        """Update detection history."""
        self.history.append(result)
        if len(self.history) > self.config.history_size:
            self.history.pop(0)
    
    def _track_pose_duration(self, pose_type: PoseType, current_time: float) -> None:
        """Track duration of current pose."""
        if pose_type != self._current_pose:
            self._current_pose = pose_type
            self._pose_start_time = current_time
    
    def _unknown_result(self) -> PoseResult:
        """Return unknown result."""
        return PoseResult(
            pose_type=PoseType.UNKNOWN,
            confidence=0.0,
            timestamp=time.time()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.history:
            return {}
        
        pose_counts = {}
        for h in self.history:
            pose = h.pose_type.value
            pose_counts[pose] = pose_counts.get(pose, 0) + 1
        
        avg_confidence = np.mean([h.confidence for h in self.history])
        
        return {
            'total_detections': len(self.history),
            'pose_distribution': pose_counts,
            'average_confidence': avg_confidence,
            'current_pose': self._current_pose.value if self._current_pose else None,
        }
    
    def reset(self) -> None:
        """Reset detector state."""
        self.history = []
        self._last_detection_time = 0.0
        self._pose_start_time = None
        self._current_pose = None
        logger.info("PoseDetector reset")
