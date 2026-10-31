"""Scene classification head and person-action head."""
import torch.nn as nn


class SceneHead(nn.Module):
    def __init__(self, input_dim, num_classes=8, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PersonActionHead(nn.Module):
    """Applied per-player, per-frame-or-final-step feature -> 9-way action logits."""

    def __init__(self, input_dim, num_classes=9, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
