"""
Neural network models for encoding and classification tasks.

This module provides optimized PyTorch models for various neural data processing tasks,
with a focus on efficient parameter usage and training speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union


class EfficientEEGClassifier(nn.Module):
    """
    An efficient convolutional neural network for EEG classification.
    
    Improvements over the original EEGClassifier:
    1. Adds dropout for regularization
    2. Uses more efficient parameter scaling
    3. Includes skip connections for better gradient flow
    4. Supports optional batch normalization
    5. Has flexible output size for multi-class classification
    """
    
    def __init__(self,
                 in_channels: int,
                 seq_len: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 num_classes: int = 2,
                 use_batchnorm: bool = True):
        """
        Initialize the classifier.
        
        Args:
            in_channels: Number of input channels (EEG electrodes)
            seq_len: Sequence length (time steps)
            hidden_size: Size of hidden layers
            num_layers: Number of convolutional layers
            dropout: Dropout probability for regularization
            num_classes: Number of output classes
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Conv1d(in_channels, hidden_size, 
                                   kernel_size=3, padding=1)
        
        # Convolutional layers with skip connections
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
            )
            self.conv_layers.append(layer)
        
        # Global pooling and output
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, L)
                B = batch size
                C = channels (EEG electrodes)
                L = sequence length (time steps)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Convolutional layers with skip connections
        for layer in self.conv_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Skip connection
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (B, hidden_size)
        x = self.dropout(x)
        
        # Output layer
        return self.fc(x)


class DualModalityModel(nn.Module):
    """
    A model that can process both EEG and fNIRS data.
    
    This model can be used for multi-modal learning by combining
    features from both EEG and fNIRS data.
    """
    
    def __init__(self,
                 eeg_channels: int,
                 eeg_seq_len: int,
                 fnirs_channels: int,
                 fnirs_seq_len: int,
                 hidden_size: int = 128,
                 fusion_type: str = "concat",
                 num_classes: int = 2):
        """
        Initialize the dual modality model.
        
        Args:
            eeg_channels: Number of EEG channels
            eeg_seq_len: EEG sequence length
            fnirs_channels: Number of fNIRS channels
            fnirs_seq_len: fNIRS sequence length
            hidden_size: Size of hidden layers
            fusion_type: How to fuse modalities ("concat", "attention", "sum")
            num_classes: Number of output classes
        """
        super().__init__()
        
        # EEG branch
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(eeg_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # fNIRS branch
        self.fnirs_encoder = nn.Sequential(
            nn.Conv1d(fnirs_channels, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fusion mechanism
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            fusion_dim = hidden_size * 2
        elif fusion_type in ("attention", "sum"):
            fusion_dim = hidden_size
            # Attention weights if needed
            if fusion_type == "attention":
                self.attention = nn.Linear(hidden_size * 2, 2)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Output layer
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with either single or dual modality.
        
        Args:
            data: Dictionary with "eeg" and/or "fnirs" tensors
                "eeg": Tensor of shape (B, eeg_channels, eeg_seq_len)
                "fnirs": Tensor of shape (B, fnirs_channels, fnirs_seq_len)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        if "eeg" in data and "fnirs" in data:
            # Both modalities available
            eeg_features = self.eeg_encoder(data["eeg"]).squeeze(-1)  # (B, hidden)
            fnirs_features = self.fnirs_encoder(data["fnirs"]).squeeze(-1)  # (B, hidden)
            
            # Fusion
            if self.fusion_type == "concat":
                features = torch.cat([eeg_features, fnirs_features], dim=1)
            elif self.fusion_type == "sum":
                features = eeg_features + fnirs_features
            elif self.fusion_type == "attention":
                # Compute attention weights
                concat = torch.cat([eeg_features, fnirs_features], dim=1)
                weights = F.softmax(self.attention(concat), dim=1)
                # Apply attention
                features = (eeg_features * weights[:, 0:1] + 
                           fnirs_features * weights[:, 1:2])
        
        elif "eeg" in data:
            # Only EEG available
            features = self.eeg_encoder(data["eeg"]).squeeze(-1)
        
        elif "fnirs" in data:
            # Only fNIRS available
            features = self.fnirs_encoder(data["fnirs"]).squeeze(-1)
        
        else:
            raise ValueError("No valid modality data provided")
        
        # Apply dropout and classify
        features = self.dropout(features)
        return self.classifier(features)


# Simplified model for rapid experimentation
class SimpleEncoder(nn.Module):
    """
    A simpler model for quick prototyping and experimentation.
    
    This model is designed for faster training and iteration.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 seq_len: int, 
                 hidden_size: int = 64,
                 num_classes: int = 2):
        """
        Initialize a simple encoder.
        
        Args:
            in_channels: Number of input channels
            seq_len: Sequence length
            hidden_size: Size of hidden layer (smaller for faster training)
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Simple convolutional encoding
            nn.Conv1d(in_channels, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, C, L)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Pass through encoder
        features = self.encoder(x).squeeze(-1)
        # Classify
        return self.classifier(features)