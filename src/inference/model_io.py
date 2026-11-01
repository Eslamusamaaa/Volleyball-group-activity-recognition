"""Load the two trained models the inference pipeline needs: the B3-A
backbone (frozen feature extractor) and the B8-mt temporal model (final
deliverable). Mirrors exactly how train_temporal.py / cache_features.py
build and load these same classes for training, so a checkpoint trained
there loads here with no surprises.
"""
import os
import torch

from ..models.backbone import ResNetBackbone, FEATURE_DIM
from ..models.temporal import PersonLSTM  # noqa: F401 (re-exported for callers)
from ..training.train_temporal import B7B8Model

ROOT = r"E:\volleyball datasets"
B3A_CKPT = os.path.join(ROOT, "outputs", "checkpoints", "person_finetune_b3a.pt")
B8MT_CKPT = os.path.join(ROOT, "outputs", "checkpoints", "b8mt_lam1.0.pt")


def load_backbone(checkpoint=B3A_CKPT, device="cpu"):
    """The B3-A backbone: ResNet50 fine-tuned on 9-way person action,
    used here purely as a frozen feature extractor (.extract_features)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    num_classes = len(ckpt["classes"])
    model = ResNetBackbone(num_classes=num_classes, mode="finetune").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_b8mt(checkpoint=B8MT_CKPT, device="cpu"):
    """B8-mt: team-separated-pooling B8 + the per-player action head, the
    final deliverable model (87.4% scene / 75.7% person-action on test)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = B7B8Model(team_pooling=True, multitask=True, input_dim=FEATURE_DIM).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
