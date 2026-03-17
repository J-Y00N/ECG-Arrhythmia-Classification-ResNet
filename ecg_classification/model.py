from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Residual 1D convolution block followed by temporal downsampling."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        pool_kernel_size: int = 5,
        pool_stride: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class ResidualCNN(nn.Module):
    """Refactored 1D residual CNN for five-class heartbeat classification."""

    def __init__(
        self,
        num_classes: int = 5,
        channels: int = 32,
        num_blocks: int = 5,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock1D(channels=channels, dropout=dropout / 2) for _ in range(num_blocks)]
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
