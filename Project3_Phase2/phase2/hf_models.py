"""Hugging Face VideoMAE model builder, wrapper, and Layer-wise Learning Rate Decay (LLRD)."""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVideoClassification,
    PreTrainedModel,
)


class DropoutLinear(nn.Linear):
    """Linear layer that applies dropout to inputs prior to linear projection.
    
    Preserves exact parameter names (weight, bias), in_features, out_features, and shapes
    so that downstream introspection and LLRD parameter grouping continue to work seamlessly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.drop_rate = drop_rate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.drop_rate > 0.0 and self.training:
            input = F.dropout(input, p=self.drop_rate, training=True)
        return F.linear(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, drop_rate: float = 0.2) -> "DropoutLinear":
        """Construct DropoutLinear by copying weights/bias from an existing Linear layer."""
        has_bias = linear.bias is not None
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            drop_rate=drop_rate,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if has_bias:
                layer.bias.copy_(linear.bias)
        return layer

from phase2.config import (
    CLASS_NAMES,
    ID2LABEL,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL2ID,
    NUM_CLASSES,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)


def verify_image_processor(model_name: str) -> Dict[str, Any]:
    """Verify that HF image processor matches Week 5 ImageNet normalization and 16 frames.
    
    Raises:
        AssertionError: If processor mean/std deviate from ImageNet constants or num_frames != 16.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    proc_mean = list(processor.image_mean)
    proc_std = list(processor.image_std)
    proc_frames = getattr(processor, "num_frames", 16)

    assert np.allclose(proc_mean, IMAGENET_MEAN, atol=1e-4), (
        f"Image processor mean {proc_mean} does not match ImageNet mean {IMAGENET_MEAN}"
    )
    assert np.allclose(proc_std, IMAGENET_STD, atol=1e-4), (
        f"Image processor std {proc_std} does not match ImageNet std {IMAGENET_STD}"
    )
    assert proc_frames == 16, f"Expected 16 frames in processor, got {proc_frames}"

    return {
        "mean": proc_mean,
        "std": proc_std,
        "num_frames": proc_frames,
    }


class VideoHFWrapper(nn.Module):
    """Thin nn.Module wrapper around Hugging Face video classification models.
    
    Standardizes the interface so forward(x) accepts (B, T, C, H, W) and returns (B, 2) logits,
    matching the Week 5 training and evaluation pipeline.
    """

    def __init__(self, hf_model: PreTrainedModel):
        super().__init__()
        self.backbone = hf_model

    @property
    def classifier(self) -> nn.Module:
        """Expose classifier head directly."""
        return getattr(self.backbone, "classifier", None)

    @property
    def config(self) -> Any:
        """Expose underlying HF config."""
        return getattr(self.backbone, "config", None)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on HF backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on HF backbone."""
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input video tensor of shape (B, T, C, H, W).
            
        Returns:
            logits: Output classification logits of shape (B, num_labels).
        """
        # Hugging Face VideoMAE expects pixel_values of shape (B, T, C, H, W)
        outputs = self.backbone(pixel_values=x)
        if hasattr(outputs, "logits"):
            return outputs.logits
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise TypeError(f"Unexpected output type from HF backbone: {type(outputs)}")


def build_hf_model(
    model_name: str = VIDEOMAE_BASE_ID,
    num_labels: int = NUM_CLASSES,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    drop_path_rate: float = 0.1,
    head_dropout: float = 0.2,
    gradient_checkpointing: bool = False,
    pretrained: bool = True,
    verify_processor: bool = True,
) -> VideoHFWrapper:
    """Build and initialize a Hugging Face video classification model with adapted 2-class head.
    
    Args:
        model_name: Hugging Face model repository ID.
        num_labels: Number of output classes (2 for binary classification).
        id2label: Mapping from class integer to class name.
        label2id: Mapping from class name to class integer.
        drop_path_rate: Drop path (stochastic depth) rate.
        head_dropout: Dropout probability applied to pooled features before classifier linear layer.
        gradient_checkpointing: Whether to enable activation checkpointing.
        pretrained: If True, loads pretrained weights; otherwise initializes from config.
        verify_processor: If True, asserts image processor matches ImageNet constants and 16 frames.
        
    Returns:
        VideoHFWrapper containing the adapted model.
    """
    if id2label is None:
        id2label = ID2LABEL
    if label2id is None:
        label2id = LABEL2ID

    if verify_processor and pretrained:
        verify_image_processor(model_name)

    # Load configuration with binary classification parameters
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        drop_path_rate=drop_path_rate,
    )

    # Instantiate model with freshly initialized head (via ignore_mismatched_sizes=True)
    if pretrained:
        raw_model = AutoModelForVideoClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    else:
        raw_model = AutoModelForVideoClassification.from_config(config)

    # Replace classifier with DropoutLinear to enforce head dropout during training
    if head_dropout > 0.0 and hasattr(raw_model, "classifier") and isinstance(raw_model.classifier, nn.Linear):
        raw_model.classifier = DropoutLinear.from_linear(raw_model.classifier, drop_rate=head_dropout)

    # Activation checkpointing
    if gradient_checkpointing:
        raw_model.gradient_checkpointing_enable()
    else:
        raw_model.gradient_checkpointing_disable()

    wrapper = VideoHFWrapper(raw_model)
    return wrapper


def get_llrd_param_groups(
    model: nn.Module,
    base_lr: float = 5e-5,
    head_lr_mult: float = 10.0,
    layer_decay: float = 0.75,
    weight_decay: float = 0.05,
) -> List[Dict[str, Any]]:
    """Construct parameter groups with Layer-wise Learning Rate Decay (LLRD).
    
    Rules:
    - Classifier head: base_lr * head_lr_mult
    - Final norm & top transformer block (layer L-1): base_lr * (layer_decay ** 0) = base_lr
    - Intermediate transformer blocks (layer l, 0 <= l < L): base_lr * (layer_decay ** (L - 1 - l))
    - Embeddings / patch projection / position: base_lr * (layer_decay ** L)
    - Bias terms, 1D parameters, LayerNorms, position embeddings: weight_decay = 0.0
    - Other 2D/3D weights: weight_decay = weight_decay
    
    The returned parameter groups are ordered from classifier down to embeddings, ensuring
    monotonically decreasing learning rates toward the input.
    
    Args:
        model: VideoHFWrapper or raw VideoMAE model.
        base_lr: Base learning rate for the top transformer block.
        head_lr_mult: Multiplier for classifier head learning rate.
        layer_decay: Multiplicative decay factor per layer depth (default 0.75).
        weight_decay: L2 regularization strength for weight matrices.
        
    Returns:
        List of parameter group dictionaries compatible with torch.optim.Optimizer.
    """
    raw_model = getattr(model, "backbone", model)

    # Determine number of transformer layers in encoder
    if hasattr(raw_model, "videomae") and hasattr(raw_model.videomae, "encoder"):
        num_layers = len(raw_model.videomae.encoder.layer)
    else:
        num_layers = getattr(raw_model.config, "num_hidden_layers", 12)

    # Collect parameters into (level, lr, is_no_decay) buckets
    # Levels:
    #   num_layers: classifier head (level L)
    #   num_layers - 1: top encoder block + fc_norm (level L-1)
    #   ...
    #   0: lowest encoder block (level 0)
    #   -1: embeddings / patch projection (level -1)
    bucket_map: Dict[Tuple[int, float, bool], List[nn.Parameter]] = {}

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue

        # Check for bias, 1D params, LayerNorm, or position embeddings -> no weight decay
        name_lower = name.lower()
        is_no_decay = (
            param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name_lower
            or "position_embeddings" in name_lower
            or "pos_embed" in name_lower
        )

        if "classifier" in name:
            level = num_layers
            lr = base_lr * head_lr_mult
        elif "fc_norm" in name:
            level = num_layers - 1
            lr = base_lr
        elif "encoder.layer." in name or "encoder.layers." in name:
            match = re.search(r"encoder\.layers?\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                level = layer_idx
                decay_power = (num_layers - 1) - layer_idx
                lr = base_lr * (layer_decay ** decay_power)
            else:
                level = num_layers - 1
                lr = base_lr
        elif "embeddings" in name:
            level = -1
            decay_power = num_layers
            lr = base_lr * (layer_decay ** decay_power)
        else:
            level = -1
            decay_power = num_layers
            lr = base_lr * (layer_decay ** decay_power)

        key = (level, lr, is_no_decay)
        if key not in bucket_map:
            bucket_map[key] = []
        bucket_map[key].append(param)

    # Sort groups by level descending (highest level / classifier first, down to embeddings)
    # Secondary key: decay first (False), no_decay second (True)
    sorted_keys = sorted(bucket_map.keys(), key=lambda k: (-k[0], k[2]))

    param_groups: List[Dict[str, Any]] = []
    for level, lr, is_no_decay in sorted_keys:
        params = bucket_map[(level, lr, is_no_decay)]
        if not params:
            continue
        wd = 0.0 if is_no_decay else weight_decay
        tag = f"level_{level}_{'nodecay' if is_no_decay else 'decay'}"
        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": wd,
            "level": level,
            "is_no_decay": is_no_decay,
            "name": tag,
        })

    # Sanity checks: total param count must exactly match trainable parameters
    trainable_numel = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    grouped_numel = sum(p.numel() for g in param_groups for p in g["params"])
    assert trainable_numel == grouped_numel, (
        f"LLRD parameter grouping lost parameters: {trainable_numel} vs {grouped_numel}"
    )

    return param_groups
