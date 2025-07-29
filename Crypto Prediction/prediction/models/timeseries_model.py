# models/timeseries_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTST(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_heads,
                 patch_size, window_size, num_classes,
                 dropout=0.0, pooling_type='cls', mlp_hidden_mult=2,
                 activation='relu', stride=None,
                 loss_type='ce', focal_gamma=2.0):

        super().__init__()

        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_loss = None 
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.pooling_type = pooling_type.lower()

        assert window_size >= patch_size, "window_size must be >= patch_size"
        assert (window_size - patch_size) % self.stride == 0, "patches must align evenly"

        self.num_patches = 1 + (window_size - patch_size) // self.stride
        self.input_proj = nn.Linear(input_size * patch_size, d_model)

        if self.pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.pos_embedding = nn.Parameter(torch.randn(
            1, self.num_patches + (1 if self.pooling_type == 'cls' else 0), d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.act = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }[activation]

        hidden_dim = d_model * mlp_hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = None
    def forward(self, x, labels=None):
        B, W, D = x.shape
        P = self.patch_size
        S = self.stride
    
        # ① Create sliding patches: (B, num_patches, patch_size * D)
        x = x.unfold(dimension=1, size=P, step=S)
        x = x.contiguous().view(B, -1, P * D)
    
        # ② Linear projection
        x = self.input_proj(x)
    
        # ③ Insert CLS token (optional)
        if self.pooling_type == 'cls':
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
    
        # ④ Add position embedding
        x = x + self.pos_embedding[:, :x.size(1), :]
    
        # ⑤ Pass through Transformer encoder
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
            if self.loss_type == 'focal':
                if self.focal_loss is None:
                    self.focal_loss = FocalLoss(gamma=self.focal_gamma)
                loss = self.focal_loss(logits, labels)
            else:
                loss = self.ce_loss(logits, labels)
            return {"loss": loss, "logits": logits}
    
        return {"logits": logits}

def load_model_from_checkpoint(path: str) -> PatchTST:
    """
    저장된 모델 체크포인트에서 PatchTST 모델을 로드합니다.

    Args:
        path (str): .pt 파일 경로

    Returns:
        PatchTST 모델 인스턴스
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model_args = checkpoint["model_args"]
    model = PatchTST(**model_args)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def load_model(**model_args) -> PatchTST:
    return PatchTST(**model_args)

class FocalLoss(torch.nn.Module):
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
