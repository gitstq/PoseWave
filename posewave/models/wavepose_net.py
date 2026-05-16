"""
WavePoseNet - Deep Learning Model for WiFi CSI Pose Detection
A lightweight CNN-LSTM architecture for real-time pose classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for WavePoseNet model."""
    num_subcarriers: int = 64
    num_antennas: int = 3
    sequence_length: int = 100
    num_classes: int = 6  # standing, sitting, lying, walking, falling, empty
    hidden_size: int = 128
    num_lstm_layers: int = 2
    dropout: float = 0.3
    cnn_channels: Tuple[int, ...] = (32, 64, 128)


class WavePoseNet(nn.Module):
    """
    WavePoseNet: CNN-LSTM Network for WiFi CSI Pose Detection.
    
    Architecture:
        - CNN encoder for spatial feature extraction from CSI data
        - LSTM for temporal modeling
        - Fully connected classifier
    
    The model processes WiFi CSI amplitude and phase data to
    classify human poses without requiring cameras.
    
    Attributes:
        config: Model configuration
        cnn_encoder: CNN feature extractor
        lstm: Temporal modeling layer
        classifier: Output classification layer
    
    Example:
        >>> model = WavePoseNet()
        >>> csi_input = torch.randn(1, 64, 3, 100)  # batch, subcarriers, antennas, time
        >>> output = model(csi_input)
        >>> predicted_pose = output.argmax(dim=1)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize WavePoseNet.
        
        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # CNN encoder for spatial features
        self.cnn_encoder = self._build_cnn_encoder()
        
        # Compute CNN output size
        self.cnn_output_size = self._compute_cnn_output_size()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_lstm_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(
            self.config.hidden_size * 2,  # Bidirectional
            1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.num_classes)
        )
        
        self._initialize_weights()
        logger.info(f"WavePoseNet initialized with {self.count_parameters()} parameters")
    
    def _build_cnn_encoder(self) -> nn.Module:
        """Build CNN encoder for CSI feature extraction."""
        layers = []
        in_channels = self.config.num_antennas
        
        for out_channels in self.config.cnn_channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(self.config.dropout)
            ])
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _compute_cnn_output_size(self) -> int:
        """Compute the output size of CNN encoder."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.config.num_antennas, self.config.sequence_length)
            output = self.cnn_encoder(dummy_input)
            return output.view(1, -1).size(1)
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through WavePoseNet.
        
        Args:
            x: Input tensor of shape (batch, subcarriers, antennas, time)
            return_attention: Whether to return attention weights
        
        Returns:
            Class logits of shape (batch, num_classes)
            Optionally returns attention weights if return_attention=True
        """
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch * subcarriers, antennas, time)
        x = x.view(-1, self.config.num_antennas, self.config.sequence_length)
        
        # CNN encoding
        cnn_features = self.cnn_encoder(x)  # (batch * subcarriers, channels, time')
        
        # Reshape for LSTM
        cnn_features = cnn_features.view(
            batch_size, 
            self.config.num_subcarriers, 
            -1
        )
        
        # LSTM encoding
        lstm_out, _ = self.lstm(cnn_features)  # (batch, subcarriers, hidden*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attention_weights.squeeze(-1)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with confidence scores.
        
        Args:
            x: Input tensor
        
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            confidence, predictions = probs.max(dim=-1)
        
        return predictions, confidence
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_pretrained(
        cls, 
        model_path: str, 
        config: Optional[ModelConfig] = None,
        device: str = 'cpu'
    ) -> 'WavePoseNet':
        """
        Load pretrained model from file.
        
        Args:
            model_path: Path to model weights
            config: Model configuration
            device: Device to load model on
        
        Returns:
            Loaded WavePoseNet model
        """
        model = cls(config)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        logger.info(f"Loaded pretrained model from {model_path}")
        return model
    
    def save(self, model_path: str) -> None:
        """Save model weights to file."""
        torch.save(self.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")


class PoseLoss(nn.Module):
    """
    Combined loss for pose detection.
    
    Combines cross-entropy loss with optional:
        - Label smoothing
        - Class weighting for imbalanced data
        - Focal loss for hard examples
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        use_focal: bool = False
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss."""
        if self.use_focal:
            loss = self._focal_loss(logits, targets)
        else:
            loss = F.cross_entropy(
                logits, 
                targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing
            )
        
        return loss
    
    def _focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = focal_loss * weights
        
        return focal_loss.mean()
