# models/timeseries_model.py
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    """
    PatchTST: Transformer-based time-series model using sliding window and patching.

    Args:
        input_size (int)
        d_model (int)
        num_layers (int)
        num_heads (int)
        patch_size (int)
        window_size (int)
        num_classes (int)
    """
    def __init__(self, input_size, d_model, num_layers, num_heads, patch_size, window_size, num_classes):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        if window_size % patch_size != 0:
            raise ValueError("window_size must be divisible by patch_size")
        self.num_patches = window_size // patch_size

        self.input_proj = nn.Linear(input_size * patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        """
        Args:
            x: shape (batch_size, window_size, input_size)
            labels (optional): ground truth labels
        Returns:
            dict: {"loss": loss, "logits": logits}
        """
        batch_size, _, input_size = x.shape
        x = x.reshape(batch_size, self.num_patches, self.patch_size * input_size)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.fc(x)
        loss = self.loss_fct(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}

def load_model(input_size, d_model, num_layers, num_heads, window_size, num_classes, patch_size):
    """
    Instantiate a PatchTST model with given parameters.
    """
    return PatchTST(input_size, d_model, num_layers, num_heads, patch_size, window_size, num_classes)
