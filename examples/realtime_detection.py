"""
PoseWave Example: Real-time Detection with WebSocket
Demonstrates real-time pose detection streaming.
"""

import asyncio
import numpy as np
import json
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def simulate_realtime_detection():
    """
    Simulate real-time pose detection.
    
    This example shows how to use PoseWave for continuous
    pose detection with WebSocket streaming.
    """
    from posewave.core.csi_processor import CSIProcessor
    from posewave.core.pose_detector import PoseDetector
    
    print("=" * 60)
    print("🌊 PoseWave Real-time Detection Simulation")
    print("=" * 60)
    
    processor = CSIProcessor()
    detector = PoseDetector()
    
    poses = ["standing", "walking", "sitting", "lying", "empty"]
    
    print("\n📡 Starting real-time detection stream...")
    print("Press Ctrl+C to stop\n")
    
    try:
        frame = 0
        while True:
            # Simulate CSI data arrival
            pose_idx = frame % len(poses)
            current_pose = poses[pose_idx]
            
            # Generate synthetic CSI
            csi_data = generate_csi_for_pose(current_pose)
            
            # Process and detect
            features = processor.process(csi_data)
            result = detector.detect(features)
            
            # Format output
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            output = {
                "timestamp": timestamp,
                "frame": frame,
                "detected_pose": result.pose_type.value,
                "confidence": f"{result.confidence:.1%}",
                "velocity": f"{result.velocity:.2f} m/s" if result.velocity else "N/A",
                "alerts": result.alerts
            }
            
            # Print result
            alert_str = f" ⚠️ {result.alerts[0]}" if result.alerts else ""
            print(f"[{timestamp}] Frame {frame:4d} | "
                  f"Pose: {result.pose_type.value:8s} | "
                  f"Conf: {result.confidence:.1%} | "
                  f"Vel: {result.velocity:.2f} m/s" if result.velocity else ""
                  f"{alert_str}")
            
            # Simulate frame rate
            await asyncio.sleep(0.1)  # 10 FPS
            frame += 1
            
            # Change pose every 30 frames
            if frame % 30 == 0:
                print(f"\n  → Simulating pose change...\n")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Detection stopped by user")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)
    
    stats = detector.get_statistics()
    print(f"  Total frames processed: {stats['total_detections']}")
    print(f"  Average confidence: {stats['average_confidence']:.1%}")
    print(f"\n  Pose Distribution:")
    for pose, count in stats['pose_distribution'].items():
        pct = count / stats['total_detections'] * 100
        bar = "█" * int(pct / 5)
        print(f"    {pose:10s}: {count:3d} ({pct:5.1f}%) {bar}")


def generate_csi_for_pose(pose: str) -> np.ndarray:
    """Generate CSI data for specific pose."""
    np.random.seed(int(datetime.now().timestamp() * 1000) % 10000)
    
    num_subcarriers = 64
    num_antennas = 3
    num_samples = 100
    
    base = np.random.randn(num_subcarriers, num_antennas, num_samples)
    
    if pose == "standing":
        base += 0.3 * np.sin(np.linspace(0, 4*np.pi, num_samples))
    elif pose == "sitting":
        base *= 0.7
    elif pose == "lying":
        base *= 0.3
    elif pose == "walking":
        base += 0.8 * np.sin(np.linspace(0, 10*np.pi, num_samples))
        base *= 1.5
    elif pose == "empty":
        base *= 0.1
    
    return base.astype(np.float32)


if __name__ == "__main__":
    asyncio.run(simulate_realtime_detection())
