"""
PoseWave Example: Basic Usage
Demonstrates the core functionality of PoseWave.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from posewave.core.csi_processor import CSIProcessor, CSIConfig
from posewave.core.pose_detector import PoseDetector, DetectorConfig
from posewave.core.signal_filter import SignalFilter


def generate_synthetic_csi(
    pose_type: str = "standing",
    num_samples: int = 100
) -> np.ndarray:
    """
    Generate synthetic CSI data for testing.
    
    Args:
        pose_type: Type of pose to simulate
        num_samples: Number of time samples
    
    Returns:
        Synthetic CSI data array
    """
    np.random.seed(42)
    
    num_subcarriers = 64
    num_antennas = 3
    
    # Base signal
    base = np.random.randn(num_subcarriers, num_antennas, num_samples)
    
    # Add pose-specific characteristics
    if pose_type == "standing":
        # Moderate variance, some periodic component
        base += 0.3 * np.sin(np.linspace(0, 4*np.pi, num_samples))
        
    elif pose_type == "sitting":
        # Lower variance
        base *= 0.7
        
    elif pose_type == "lying":
        # Very low variance
        base *= 0.3
        
    elif pose_type == "walking":
        # High variance, strong periodic component
        base += 0.8 * np.sin(np.linspace(0, 10*np.pi, num_samples))
        base *= 1.5
        
    elif pose_type == "empty":
        # Very low variance
        base *= 0.1
    
    return base.astype(np.float32)


def main():
    """Run basic usage example."""
    print("=" * 60)
    print("🌊 PoseWave - WiFi CSI Human Pose Detection Example")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Initializing components...")
    csi_config = CSIConfig(
        num_subcarriers=64,
        num_antennas=3,
        sample_rate=100.0
    )
    detector_config = DetectorConfig(
        confidence_threshold=0.7,
        enable_alerts=True
    )
    
    processor = CSIProcessor(csi_config)
    detector = PoseDetector(detector_config)
    signal_filter = SignalFilter()
    
    print("✅ CSI Processor initialized")
    print("✅ Pose Detector initialized")
    print("✅ Signal Filter initialized")
    
    # Test different poses
    poses = ["standing", "sitting", "lying", "walking", "empty"]
    
    print("\n" + "=" * 60)
    print("🔍 Testing Pose Detection")
    print("=" * 60)
    
    for pose in poses:
        print(f"\n📍 Simulating: {pose.upper()}")
        print("-" * 40)
        
        # Generate synthetic CSI data
        csi_data = generate_synthetic_csi(pose, num_samples=100)
        
        # Process CSI data
        features = processor.process(csi_data)
        
        # Detect pose
        result = detector.detect(features)
        
        # Print results
        print(f"  Detected Pose: {result.pose_type.value}")
        print(f"  Confidence:    {result.confidence:.2%}")
        print(f"  Velocity:      {result.velocity:.2f} m/s" if result.velocity else "  Velocity:      N/A")
        
        if result.alerts:
            print(f"  ⚠️  Alerts:     {', '.join(result.alerts)}")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("📊 Detection Statistics")
    print("=" * 60)
    
    stats = detector.get_statistics()
    print(f"  Total Detections: {stats['total_detections']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2%}")
    print(f"  Pose Distribution:")
    for pose, count in stats['pose_distribution'].items():
        print(f"    - {pose}: {count}")
    
    # Test signal filtering
    print("\n" + "=" * 60)
    print("🔧 Signal Filtering Demo")
    print("=" * 60)
    
    noisy_signal = np.random.randn(100) + 0.5 * np.sin(np.linspace(0, 4*np.pi, 100))
    
    filtered_bandpass = signal_filter.filter(noisy_signal, filter_type="bandpass")
    filtered_kalman = signal_filter.filter(noisy_signal, filter_type="kalman")
    smoothed = signal_filter.moving_average(noisy_signal, window=5)
    
    print(f"  Original signal variance:    {np.var(noisy_signal):.4f}")
    print(f"  Bandpass filtered variance:  {np.var(filtered_bandpass):.4f}")
    print(f"  Kalman filtered variance:    {np.var(filtered_kalman):.4f}")
    print(f"  Moving average variance:     {np.var(smoothed):.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
