"""Tests verifying scratch models unfreeze BatchNorm and train running stats."""

import pytest
import torch
import torch.nn as nn

from phase2.train_scratch import ScratchResNetBiGRU, ScratchVideoCNN3D


def test_scratch_resnet_bigru_trainable_bn():
    """Verify ScratchResNetBiGRU keeps all parameters and BatchNorm trainable during train()."""
    model = ScratchResNetBiGRU(num_classes=2, hidden_dim=64, gru_layers=1)
    
    # Check all parameters have requires_grad == True
    for name, p in model.named_parameters():
        assert p.requires_grad, f"Parameter {name} in ScratchResNetBiGRU is frozen!"

    model.train()
    # Check that BatchNorm2d layers are in training mode
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    assert len(bn_layers) > 0, "No BatchNorm layers found!"
    for bn in bn_layers:
        assert bn.training, "BatchNorm layer was forced to eval mode in train()!"
        for p in bn.parameters():
            assert p.requires_grad, "BatchNorm affine parameter was frozen!"

    # Forward pass smoke test
    x = torch.randn(2, 16, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)


def test_scratch_video_cnn3d_trainable_bn():
    """Verify ScratchVideoCNN3D keeps all parameters and BatchNorm trainable during train()."""
    model = ScratchVideoCNN3D(num_classes=2)

    # Check all parameters have requires_grad == True
    for name, p in model.named_parameters():
        assert p.requires_grad, f"Parameter {name} in ScratchVideoCNN3D is frozen!"

    model.train()
    # Check that BatchNorm3d layers are in training mode
    bn3d_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d))]
    assert len(bn3d_layers) > 0, "No BatchNorm layers found!"
    for bn in bn3d_layers:
        assert bn.training, "BatchNorm layer was forced to eval mode in train()!"
        for p in bn.parameters():
            assert p.requires_grad, "BatchNorm affine parameter was frozen!"

    # Forward pass smoke test
    x = torch.randn(2, 16, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2)
