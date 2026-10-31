"""LSTM1 (per-player), LSTM2 (scene-level), and pooling variants for B4-B8."""
import torch
import torch.nn as nn


class FrameLSTM(nn.Module):
    """Single LSTM over a sequence of feature vectors -> final hidden state. Used for B4/B6 (image-level) and as LSTM2 (scene-level)."""

    def __init__(self, input_dim, hidden_dim=512, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        # x: (B, T, input_dim) -> returns (B, T, out_dim) full sequence and (B, out_dim) final step
        seq, _ = self.lstm(x)
        return seq, seq[:, -1, :]


class PersonLSTM(nn.Module):
    """LSTM1: one LSTM applied independently per player across the time axis.

    Input (B, P, T, D) -> output full sequence (B, P, T, H) and final-step (B, P, H).
    Players share LSTM weights (same model applied per player).
    """

    def __init__(self, input_dim, hidden_dim=512, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        B, P, T, D = x.shape
        flat = x.reshape(B * P, T, D)
        seq, _ = self.lstm(flat)
        seq = seq.reshape(B, P, T, self.out_dim)
        return seq, seq[:, :, -1, :]


def masked_max_pool(x, mask):
    """x: (B, P, H) player features, mask: (B, P) bool -> (B, H) pooled over valid players."""
    x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = x.max(dim=1).values
    pooled = torch.nan_to_num(pooled, neginf=0.0)  # clips with 0 valid players (shouldn't happen)
    return pooled


def team_pool(x, mask, team_mask):
    """Group-style pooling (B8): max-pool each team separately, then concatenate.

    x: (B, P, H), mask: (B, P) bool, team_mask: (B, P) long in {0, 1}
    returns (B, 2H)
    """
    team0 = mask & (team_mask == 0)
    team1 = mask & (team_mask == 1)
    x0 = masked_max_pool(x, team0)
    x1 = masked_max_pool(x, team1)
    return torch.cat([x0, x1], dim=-1)
