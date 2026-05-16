"""
WebSocket Manager
Real-time pose detection streaming via WebSocket.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import asyncio
import json
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket Connection Manager for real-time pose streaming.
    
    Manages WebSocket connections and broadcasts pose detection
    results to connected clients in real-time.
    
    Example:
        >>> manager = ConnectionManager()
        >>> # In WebSocket endpoint:
        >>> await manager.connect(websocket)
        >>> await manager.broadcast({"pose": "standing", "confidence": 0.95})
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        self._stats = {
            'total_messages': 0,
            'total_connections': 0,
            'start_time': time.time()
        }
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            self._stats['total_connections'] += 1
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_personal_message(
        self, 
        message: Dict[str, Any], 
        websocket: WebSocket
    ) -> None:
        """
        Send message to a specific client.
        
        Args:
            message: Message dictionary to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
            self._stats['total_messages'] += 1
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message dictionary to broadcast
        """
        if not self.active_connections:
            return
        
        message_json = json.dumps(message, default=str)
        disconnected = []
        
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_json)
                    self._stats['total_messages'] += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            await self.disconnect(conn)
    
    async def broadcast_pose(
        self,
        pose: str,
        confidence: float,
        velocity: Optional[float] = None,
        alerts: Optional[List[str]] = None
    ) -> None:
        """
        Broadcast pose detection result.
        
        Args:
            pose: Detected pose type
            confidence: Detection confidence
            velocity: Optional velocity estimate
            alerts: Optional list of alerts
        """
        message = {
            'type': 'pose_detection',
            'timestamp': time.time(),
            'data': {
                'pose': pose,
                'confidence': confidence,
                'velocity': velocity,
                'alerts': alerts or []
            }
        }
        await self.broadcast(message)
    
    async def broadcast_features(self, features: Dict[str, Any]) -> None:
        """
        Broadcast extracted features.
        
        Args:
            features: Feature dictionary
        """
        message = {
            'type': 'features',
            'timestamp': time.time(),
            'data': {
                'amplitude_mean': float(np.mean(features.get('amplitude', []))),
                'amplitude_var': float(np.var(features.get('amplitude', []))),
                'doppler_mean': float(np.mean(features.get('doppler', []))),
            }
        }
        await self.broadcast(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime = time.time() - self._stats['start_time']
        return {
            'active_connections': len(self.active_connections),
            'total_connections': self._stats['total_connections'],
            'total_messages': self._stats['total_messages'],
            'uptime_seconds': uptime
        }


async def websocket_endpoint(
    websocket: WebSocket,
    manager: ConnectionManager
):
    """
    WebSocket endpoint handler for pose streaming.
    
    Handles incoming CSI data and streams pose detection results.
    
    Args:
        websocket: WebSocket connection
        manager: Connection manager instance
    """
    await manager.connect(websocket)
    
    try:
        # Import here to avoid circular imports
        from posewave.api.main import get_csi_processor, get_pose_detector
        
        processor = get_csi_processor()
        detector = get_pose_detector()
        
        while True:
            # Receive CSI data
            data = await websocket.receive_json()
            
            # Process CSI data
            if 'csi_data' in data:
                csi_array = np.array(data['csi_data'])
                features = processor.process(csi_array)
                result = detector.detect(features)
                
                # Send back detection result
                await manager.send_personal_message({
                    'type': 'pose_result',
                    'pose': result.pose_type.value,
                    'confidence': result.confidence,
                    'velocity': result.velocity,
                    'alerts': result.alerts,
                    'timestamp': result.timestamp
                }, websocket)
            
            # Handle commands
            elif 'command' in data:
                command = data['command']
                
                if command == 'get_stats':
                    await manager.send_personal_message({
                        'type': 'stats',
                        'data': manager.get_stats()
                    }, websocket)
                
                elif command == 'get_detection_stats':
                    await manager.send_personal_message({
                        'type': 'detection_stats',
                        'data': detector.get_statistics()
                    }, websocket)
                
                elif command == 'reset':
                    detector.reset()
                    processor.reset()
                    await manager.send_personal_message({
                        'type': 'status',
                        'message': 'Reset successful'
                    }, websocket)
    
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info("Client disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
