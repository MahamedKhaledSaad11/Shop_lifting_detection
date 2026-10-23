"""3D Spatiotemporal CNN (R(2+1)D-18) for video classification."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class VideoCNN3D(nn.Module):
    """Spatiotemporal 3D CNN based on factorized R(2+1)D-18 pretrained on Kinetics-400.
    
    Accepts (B, T, C, H, W) video tensors from ShopliftingDataset and internally
    permutes to (B, C, T, H, W) for 3D spatiotemporal convolutions.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        head_dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Kinetics-400 pretrained weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        model = r2plus1d_18(weights=weights)

        in_features = model.fc.in_features  # 512
        # Replace 400-class Kinetics head with new regularized 2-class head
        model.fc = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W)
            
        Returns:
            logits: Output classification tensor of shape (B, num_classes)
        """
        # Permute from dataset layout (B, T, C, H, W) -> 3D CNN layout (B, C, T, H, W)
        x_3d = x.permute(0, 2, 1, 3, 4)
        logits = self.model(x_3d)
        return logits

    def train(self, mode: bool = True):
        """Set training mode while strictly enforcing Frozen-BN on 3D conv backbone."""
        super().train(mode)
        if mode:
            # Enforce Frozen BatchNorm3d to preserve Kinetics features and prevent small-batch degradation
            for name, m in self.model.named_modules():
                if not name.startswith("fc") and isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

            # Keep any backbone submodule whose parameters are all frozen in eval mode
            for name, child in self.model.named_children():
                if name != "fc" and not any(p.requires_grad for p in child.parameters()):
                    child.eval()
        return self

    def freeze_backbone(self) -> None:
        """Freeze 3D CNN backbone for warmup stage."""
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        for name, child in self.model.named_children():
            if name != "fc":
                child.eval()
        self.model.fc.train()

    def unfreeze_backbone(self, layers: Optional[List[str]] = None) -> None:
        """Unfreeze deeper 3D convolutional blocks (default: layer3 and layer4) while keeping BN frozen."""
        if layers is None:
            layers = ["layer3", "layer4"]

        for name, child in self.model.named_children():
            if any(l in name for l in layers) or name == "fc":
                for param in child.parameters():
                    param.requires_grad = True
                child.train()
            else:
                for param in child.parameters():
                    param.requires_grad = False
                child.eval()

        # Re-enforce Frozen-BN
        for name, m in self.model.named_modules():
            if not name.startswith("fc") and isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
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

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("fc"):
                head_params.append(p)
            else:
                backbone_params.append(p)

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
                "name": "backbone_3d",
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
                "name": "head_3d",
            })
        return groups
