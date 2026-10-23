"""ResNet + BiGRU with Explicit Motion Deltas and Hybrid Temporal Self-Attention Pooling."""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class TemporalSelfAttentionPooling(nn.Module):
    """Learnable Temporal Self-Attention pooling over sequence hidden states.
    
    Computes a normalized attention distribution over time:
        e_t = w^T tanh(W h_t + b)
        alpha = softmax(e)
        c_attn = sum_t alpha_t h_t
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Tensor of shape (B, T, hidden_dim)
            
        Returns:
            context: Tensor of shape (B, hidden_dim)
            weights: Tensor of shape (B, T, 1)
        """
        # (B, T, 1)
        scores = self.projection(h)
        weights = F.softmax(scores, dim=1)
        # Context vector: sum over time (B, hidden_dim)
        context = torch.sum(weights * h, dim=1)
        return context, weights


class ResNetBiGRUAttention(nn.Module):
    """Video classification network combining:
    1. ImageNet-pretrained ResNet backbone per frame.
    2. Explicit temporal motion deltas (Delta f_t = f_t - f_{t-1}) canceling static CCTV background.
    3. 2-layer Bidirectional GRU for temporal aggregation.
    4. Hybrid temporal pooling (Attention + Mean residual) to focus on sparse theft actions.
    5. Regularized classification head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 1. 2D Spatial Backbone (ResNet-18)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet-18

        # Remove original classification fc and keep spatial feature pipeline
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # 2. Motion Delta Fusion: [f_t, Delta f_t] -> 2 * feature_dim (1024)
        in_dim = 2 * self.feature_dim
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_dropout = nn.Dropout(0.2)

        # 3. Bidirectional GRU
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = 2 * hidden_dim  # bidirectional -> 512

        # 4. Hybrid Temporal Pooling (Attention + Global Mean Residual)
        self.attention = TemporalSelfAttentionPooling(gru_out_dim, attention_dim=128)
        self.pool_norm = nn.LayerNorm(gru_out_dim)

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract ResNet features per frame.
        
        Args:
            x: (B, T, C, H, W)
            
        Returns:
            features: (B, T, feature_dim)
        """
        B, T, C, H, W = x.shape
        # Fold time into batch: (B*T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        feats_flat = self.backbone(x_flat)  # (B*T, 512)
        feats = feats_flat.view(B, T, self.feature_dim)
        return feats

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Any:
        """Forward pass.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            return_attention: If True, also returns attention weights (B, T, 1)
            
        Returns:
            logits: (B, num_classes)
            (optional) attn_weights: (B, T, 1)
        """
        B, T, C, H, W = x.shape

        # 1. Per-frame spatial feature extraction
        f = self.extract_features(x)  # (B, T, D)

        # 2. Explicit Temporal Motion Deltas (canceling ~90% static CCTV background)
        delta_f = torch.zeros_like(f)
        if T > 1:
            delta_f[:, 1:, :] = f[:, 1:, :] - f[:, :-1, :]

        # Concatenate appearance f_t and kinematic delta Delta f_t
        f_combined = torch.cat([f, delta_f], dim=-1)  # (B, T, 2D)
        f_normed = self.input_dropout(self.input_norm(f_combined))

        # 3. Temporal modeling with BiGRU
        h, _ = self.gru(f_normed)  # (B, T, 2 * hidden_dim)

        # 4. Hybrid Temporal Pooling (Attention + Global Mean Residual)
        c_attn, attn_weights = self.attention(h)  # (B, 512), (B, T, 1)
        c_mean = torch.mean(h, dim=1)             # (B, 512)
        c = self.pool_norm(c_attn + c_mean)       # (B, 512)

        # 5. Classification Logits
        logits = self.classifier(c)  # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    def train(self, mode: bool = True):
        """Set training mode while strictly enforcing Frozen-BN and frozen backbone submodule eval mode."""
        super().train(mode)
        if mode:
            # Enforce Frozen BatchNorm2d to prevent small-batch statistics degradation on CCTV video
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

            # Keep any backbone submodule whose parameters are all frozen in eval mode
            for child in self.backbone.children():
                if not any(p.requires_grad for p in child.parameters()):
                    child.eval()
        return self

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the 2D CNN backbone for warmup stage."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self, layers: Optional[List[str]] = None) -> None:
        """Unfreeze specified deeper backbone layers (default: layer3 and layer4) while keeping BN frozen."""
        if layers is None:
            layers = ["layer3", "layer4"]

        for name, child in self.backbone.named_children():
            if any(l in name for l in layers):
                for param in child.parameters():
                    param.requires_grad = True
                child.train()
            else:
                for param in child.parameters():
                    param.requires_grad = False
                child.eval()

        # Re-enforce Frozen-BN on all backbone modules
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def get_param_groups(
        self,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Return differential learning rate parameter groups for optimizer."""
        backbone_params = []
        head_params = []

        backbone_ids = {id(p) for p in self.backbone.parameters()}

        for p in self.parameters():
            if not p.requires_grad:
                continue
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                head_params.append(p)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "name": "backbone",
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head_gru",
            })
        return groups
