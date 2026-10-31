"""Loss functions for severe class imbalance (Section 3.6: jumping is ~100x
rarer than standing)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss: down-weights easy (already well-classified)
    examples via (1-p_t)^gamma, on top of optional static class weights.
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_balanced_sampler(labels, num_classes):
    """WeightedRandomSampler with per-sample weight = 1/class_count.

    Gives every class roughly equal expected exposure per epoch: effectively
    oversamples rare classes (jumping) and undersamples the dominant one
    (standing) by resampling-with-replacement probability, without needing
    separate oversample/undersample code paths.
    """
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float()
    counts = counts.clamp(min=1)
    class_weight = 1.0 / counts
    sample_weights = class_weight[torch.as_tensor(labels)]
    return torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)
