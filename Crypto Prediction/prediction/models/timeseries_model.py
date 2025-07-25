# models/timeseries_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import numpy as np


class ModelConfig(PretrainedConfig):
    """
    Configuration class for the time series prediction model.
    Note: Specific architecture details are proprietary.
    """
    model_type = "timeseries_transformer"

    def __init__(
        self,
        n_features=8,  # Number of input features
        n_classes=2,   # Number of output classes
        window_size=14,  # Default window size
        **kwargs
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.window_size = window_size
        super().__init__(**kwargs)


class TimeSeriesTransformer(PreTrainedModel):
    """
    Time Series Transformer model for cryptocurrency prediction.
    
    Note: This is a simplified version for portfolio demonstration.
    The actual production model contains proprietary architecture details.
    """
    config_class = ModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Simplified architecture (actual implementation is proprietary)
        self.feature_projection = nn.Linear(config.n_features, 128)
        self.transformer_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        self.classifier = nn.Linear(128, config.n_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features)
        
        Returns:
            Dictionary containing logits and other outputs
        """
        # Project features
        x = self.feature_projection(x)
        
        # Apply transformer layers
        x = self.transformer_layers(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Apply dropout and classification layer
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return {"logits": logits}


# Alias for backward compatibility
PatchTST = TimeSeriesTransformer


def load_model_from_checkpoint(path: str) -> TimeSeriesTransformer:
    """
    Load TimeSeriesTransformer model from saved checkpoint.

    Args:
        path (str): Path to .pt file

    Returns:
        TimeSeriesTransformer model instance
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model_args = checkpoint["model_args"]
    
    # Create config from model args
    config = ModelConfig(**model_args)
    model = TimeSeriesTransformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def load_model(**model_args) -> TimeSeriesTransformer:
    config = ModelConfig(**model_args)
    return TimeSeriesTransformer(config)

class FocalLoss(torch.nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
