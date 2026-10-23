"""Unit tests and 1-batch overfit smoke tests for video models."""

import pytest
import torch
import torch.nn as nn

from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D


def test_resnet_bigru_attention_shapes():
    """Verify forward pass output shapes and attention weights."""
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)

    model = ResNetBiGRUAttention(num_classes=2, pretrained=False, hidden_dim=64)
    out = model(x)
    assert out.shape == (B, 2), f"Expected (B, 2), got {out.shape}"

    # Test with return_attention
    out, attn = model(x, return_attention=True)
    assert out.shape == (B, 2)
    assert attn.shape == (B, T, 1)
    # Check that attention weights sum to ~1 over time
    attn_sums = torch.sum(attn, dim=1).squeeze(-1)
    assert torch.allclose(attn_sums, torch.ones(B), atol=1e-5)


def test_video_cnn3d_shapes():
    """Verify 3D CNN forward pass accepts (B, T, C, H, W) and returns (B, 2)."""
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)

    model = VideoCNN3D(num_classes=2, pretrained=False)
    out = model(x)
    assert out.shape == (B, 2), f"Expected (B, 2), got {out.shape}"


def test_frozen_bn_invariance_under_train():
    """Verify that model.train() preserves Frozen-BN mode on backbone."""
    model_2d = ResNetBiGRUAttention(num_classes=2, pretrained=False)
    model_2d.train()  # Call train mode

    for m in model_2d.backbone.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            assert m.training is False, "BatchNorm in ResNet backbone should remain eval!"
            for p in m.parameters():
                assert p.requires_grad is False, "BN parameters must be frozen!"

    model_3d = VideoCNN3D(num_classes=2, pretrained=False)
    model_3d.train()
    for name, m in model_3d.model.named_modules():
        if not name.startswith("fc") and isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            assert m.training is False, "BatchNorm in 3D backbone should remain eval!"
            for p in m.parameters():
                assert p.requires_grad is False


def test_differential_param_groups():
    """Verify get_param_groups separates backbone from head/temporal parameters."""
    model = ResNetBiGRUAttention(num_classes=2, pretrained=False)
    groups = model.get_param_groups(lr_backbone=1e-5, lr_head=1e-4, weight_decay=1e-4)

    assert len(groups) == 2
    names = {g["name"] for g in groups}
    assert "backbone" in names
    assert "head_gru" in names

    # When backbone is frozen
    model.freeze_backbone()
    groups_frozen = model.get_param_groups(lr_backbone=1e-5, lr_head=1e-4)
    assert len(groups_frozen) == 1
    assert groups_frozen[0]["name"] == "head_gru"


def test_one_batch_overfit_smoke_test_2d():
    """Smoke test: verify that a single batch overfits ResNet+BiGRU (loss drops steeply over 25 steps)."""
    torch.manual_seed(42)
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)
    y = torch.tensor([0, 1], dtype=torch.long)

    model = ResNetBiGRUAttention(
        num_classes=2,
        pretrained=False,
        hidden_dim=64,
        head_dropout=0.0,
        gru_dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for step in range(25):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss is not None
    assert final_loss < 0.05, f"Expected final loss < 0.05, got {final_loss:.4f}"
    assert final_loss < initial_loss * 0.1, "Loss did not drop by at least 90%!"


def test_one_batch_overfit_smoke_test_3d():
    """Smoke test: verify that a single batch overfits VideoCNN3D."""
    torch.manual_seed(42)
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = torch.randn(B, T, C, H, W)
    y = torch.tensor([0, 1], dtype=torch.long)

    model = VideoCNN3D(num_classes=2, pretrained=False, head_dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    initial_loss = None
    final_loss = None

    for step in range(20):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss is not None
    assert final_loss < 0.05, f"Expected final loss < 0.05, got {final_loss:.4f}"
