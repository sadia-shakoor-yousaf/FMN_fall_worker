"""Tiny LSTM on 5 pose features (CPU-friendly)."""

from __future__ import annotations

import torch
from torch import nn


class FallLSTM(nn.Module):
    def __init__(
        self, n_feat: int = 5, hidden: int = 64, layers: int = 1, n_class: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
