from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.proj(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        return self.act(y + res)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        channels = channels or [256, 256, 256, 256]
        layers = []
        in_ch = input_dim
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) -> (B,D,T)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y.mean(dim=-1)  # global avg pooling over T
        return self.head(y)


class GRUBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])
