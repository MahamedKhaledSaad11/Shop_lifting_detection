"""Unit tests for Hugging Face VideoMAE model loading, head adaptation, gradients, LLRD, and overfitting."""

import pytest
import torch
import torch.nn as nn

from phase2.config import (
    DEVICE,
    ID2LABEL,
    LABEL2ID,
    NUM_CLASSES,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import VideoHFWrapper, build_hf_model, get_llrd_param_groups


@pytest.fixture(scope="module")
def small_model():
    """Fixture providing VideoMAE-Small."""
    return build_hf_model(
        model_name=VIDEOMAE_SMALL_ID,
        num_labels=NUM_CLASSES,
        gradient_checkpointing=False,
        verify_processor=True,
    )


class TestHFModel:
    """Test suite for Hugging Face video classification model integration."""

    def test_model_loads_and_head_adapted(self, small_model):
        """Test that model loads cleanly and classifier head has out_features == 2."""
        assert isinstance(small_model, VideoHFWrapper)
        assert small_model.classifier is not None
        assert small_model.classifier.out_features == 2
        assert small_model.config.num_labels == 2
        assert small_model.config.id2label[0] == "normal"
        assert small_model.config.id2label[1] == "shoplifting"
        assert small_model.config.label2id["normal"] == 0
        assert small_model.config.label2id["shoplifting"] == 1
        assert hasattr(small_model.classifier, "drop_rate")
        assert small_model.classifier.drop_rate == 0.2

    def test_forward_pass_logits_shape(self, small_model):
        """Test forward pass accepts (B, T, C, H, W) and outputs (B, 2) logits."""
        small_model.eval()
        B, T, C, H, W = 2, 16, 3, 224, 224
        x = torch.randn(B, T, C, H, W)
        with torch.no_grad():
            logits = small_model(x)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (B, 2)
        assert not torch.isnan(logits).any()

    def test_gradients_reach_head_and_last_encoder_block(self, small_model):
        """Test backward pass propagates non-zero gradients to both head and last encoder block."""
        small_model.train()
        small_model.zero_grad()

        B, T, C, H, W = 2, 16, 3, 224, 224
        x = torch.randn(B, T, C, H, W, requires_grad=False)
        targets = torch.tensor([0, 1], dtype=torch.long)

        logits = small_model(x)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        # Check classifier head gradient
        assert small_model.classifier.weight.grad is not None
        head_grad_norm = small_model.classifier.weight.grad.norm().item()
        assert head_grad_norm > 0.0, "Classifier head received zero gradients."

        # Check last encoder layer gradient
        raw = small_model.backbone
        last_layer = raw.videomae.encoder.layer[-1]
        last_layer_grad = last_layer.output.dense.weight.grad
        assert last_layer_grad is not None
        last_grad_norm = last_layer_grad.norm().item()
        assert last_grad_norm > 0.0, "Last encoder layer received zero gradients."

    def test_llrd_param_groups_monotonic_and_weight_decay(self, small_model):
        """Test LLRD groups have monotonically decreasing LR toward embeddings and no WD on bias/norm."""
        base_lr = 5e-5
        head_mult = 10.0
        decay = 0.75
        wd = 0.05

        param_groups = get_llrd_param_groups(
            small_model,
            base_lr=base_lr,
            head_lr_mult=head_mult,
            layer_decay=decay,
            weight_decay=wd,
        )

        assert len(param_groups) > 0

        # Check parameter count preservation
        all_trainable = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
        grouped_trainable = sum(p.numel() for g in param_groups for p in g["params"])
        assert all_trainable == grouped_trainable

        # Check weight decay policy: bias and norm params must have wd == 0.0
        for group in param_groups:
            is_no_decay = group["is_no_decay"]
            if is_no_decay:
                assert group["weight_decay"] == 0.0, (
                    f"Group {group['name']} should have weight_decay=0.0, got {group['weight_decay']}"
                )
            else:
                assert group["weight_decay"] == wd, (
                    f"Group {group['name']} should have weight_decay={wd}, got {group['weight_decay']}"
                )

        # Check monotonic decrease of learning rates from head down to embeddings
        lrs = [g["lr"] for g in param_groups]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12, (
                f"Learning rates are not monotonically decreasing: {lrs[i-1]} followed by {lrs[i]}"
            )

        # Check distinct level LRs are strictly decreasing
        unique_lrs = []
        for lr in lrs:
            if not any(abs(lr - u) < 1e-12 for u in unique_lrs):
                unique_lrs.append(lr)

        assert unique_lrs[0] == pytest.approx(base_lr * head_mult)  # Head: 5e-4
        assert unique_lrs[1] == pytest.approx(base_lr)              # Top layer: 5e-5
        for i in range(1, len(unique_lrs)):
            assert unique_lrs[i] < unique_lrs[i - 1], (
                f"Expected unique LRs to strictly decrease: {unique_lrs[i-1]} vs {unique_lrs[i]}"
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 1-batch overfit test")
    def test_one_batch_overfit_videomae_small(self):
        """Test that VideoMAE-Small can overfit 1 batch to loss < 0.05 within 40 steps."""
        torch.manual_seed(42)
        device = torch.device("cuda")

        model = build_hf_model(
            model_name=VIDEOMAE_SMALL_ID,
            num_labels=2,
            gradient_checkpointing=True,
            verify_processor=False,
        ).to(device)

        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")

        # Fixed batch of 2 distinct classes
        x = torch.randn(2, 16, 3, 224, 224, device=device)
        y = torch.tensor([0, 1], dtype=torch.long, device=device)

        final_loss = 1.0
        for step in range(40):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            final_loss = loss.item()
            if final_loss < 0.05:
                break

        assert final_loss < 0.05, f"1-batch overfit failed to reach loss < 0.05 (got {final_loss:.4f})"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 1-batch overfit test")
    def test_one_batch_overfit_videomae_base(self):
        """Test that VideoMAE-Base can overfit 1 batch to loss < 0.05 within 40 steps."""
        torch.manual_seed(42)
        device = torch.device("cuda")

        model = build_hf_model(
            model_name=VIDEOMAE_BASE_ID,
            num_labels=2,
            gradient_checkpointing=True,
            verify_processor=False,
        ).to(device)

        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda")

        # Fixed batch of 2 distinct classes
        x = torch.randn(2, 16, 3, 224, 224, device=device)
        y = torch.tensor([0, 1], dtype=torch.long, device=device)

        final_loss = 1.0
        for step in range(40):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            final_loss = loss.item()
            if final_loss < 0.05:
                break

        assert final_loss < 0.05, f"VideoMAE-B 1-batch overfit failed to reach loss < 0.05 (got {final_loss:.4f})"

