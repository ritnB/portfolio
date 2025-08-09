import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss as a replacement for CrossEntropy for imbalanced data."""
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class PatchSequenceModel(nn.Module):
    """Generic patch-based Transformer for sequence classification."""
    def __init__(self, input_size, d_model, num_layers, num_heads,
                 patch_size, window_size, num_classes,
                 dropout=0.0, pooling_type='cls', mlp_hidden_mult=2,
                 activation='relu', stride=None):

        super().__init__()

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size  # default to patch_size
        self.pooling_type = pooling_type.lower()

        assert window_size >= patch_size, "window_size must be >= patch_size"
        assert (window_size - patch_size) % self.stride == 0, "patches must align evenly"

        self.num_patches = 1 + (window_size - patch_size) // self.stride


        # Patch embedding
        self.input_proj = nn.Linear(input_size * patch_size, d_model)

        # CLS token
        if self.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(
            1, self.num_patches + (1 if self.pooling_type == 'cls' else 0), d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Activation
        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }[activation]

        # MLP Head
        hidden_dim = d_model * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = None  # can be set externally

    def forward(self, x, labels=None):
        B, W, D = x.shape
        P = self.patch_size
        S = self.stride

        # ① Create sliding patches: (B, num_patches, patch_size * D)
        x = x.unfold(dimension=1, size=P, step=S)
        x = x.contiguous().view(B, -1, P * D)

        # ② Linear projection
        x = self.input_proj(x)

        # ③ Optionally insert CLS token
        if self.pooling_type == 'cls':
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        # ④ Add positional embedding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # ⑤ Transformer encoder
        x = self.transformer(x)

        # ⑥ Pooling
        if self.pooling_type == 'cls':
            x = x[:, 0, :]
        else:
            x = x.mean(dim=1)

        # ⑦ Classification
        logits = self.mlp(x)

        # ⑧ Loss (optional)
        if labels is not None:
            if self.focal_loss is not None:
                loss = self.focal_loss(logits, labels)
            else:
                loss = self.ce_loss(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

def load_model_from_checkpoint(path: str, return_args: bool = False):
    """Load PatchTST model from a checkpoint (v11_4 compatible).

    Args:
        path (str): checkpoint path (.pt)
        return_args (bool): if True, return (model, model_args)

    Returns:
        PatchTST model instance or (model, model_args)
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model_args = checkpoint["model_args"]
    
    # v11_4 compatibility: patch_len -> patch_size
    if 'patch_len' in model_args and 'patch_size' not in model_args:
        model_args['patch_size'] = model_args['patch_len']
    
    # v11_4 compatibility: dropout_mlp -> dropout
    if 'dropout_mlp' in model_args and 'dropout' not in model_args:
        model_args['dropout'] = model_args['dropout_mlp']
    
    # Defaults for parameters added in v11_4
    model_args.setdefault('pooling_type', 'cls')
    model_args.setdefault('activation', 'relu')
    model_args.setdefault('mlp_hidden_mult', 2)
    model_args.setdefault('stride', model_args.get('patch_size', 16))
    
    # Filter to constructor-supported args
    valid_params = {
        'input_size', 'd_model', 'num_layers', 'num_heads', 
        'patch_size', 'window_size', 'num_classes', 'dropout', 
        'pooling_type', 'mlp_hidden_mult', 'activation', 'stride'
    }
    filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
    
    model = PatchSequenceModel(**filtered_args)
    model.load_state_dict(checkpoint["state_dict"])
    
    if return_args:
        return model, model_args  # return original model_args (includes v11_4 settings)
    return model

def load_model(**model_args) -> PatchSequenceModel:
    return PatchSequenceModel(**model_args)
