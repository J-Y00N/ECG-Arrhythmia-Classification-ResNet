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
    """1D residual CNN for heartbeat classification.

    Interval features, where an arm supplies them, join at the classifier rather
    than at the stem. They arrive stacked behind the waveform as constant planes,
    because that is what keeps the dataloader returning a plain ``(input, label)``
    pair and leaves the training loop, the metrics and the prediction table
    untouched; the model unpacks them on the way in.

    Two reasons for taking them apart here rather than feeding all the channels
    to the first convolution. The interval features are four scalars, not signal,
    so passing them through five rounds of convolution and pooling wastes the
    trunk on values that never vary along the axis it operates over. And leaving
    the trunk at a single input channel means the convolutional stack is
    identical in every arm, so a difference in results is attributable to what
    the classifier was given rather than to a reshaped first layer.

    With ``n_interval_features = 0`` the forward pass and the parameter names are
    the same as the version that preceded the interval arms, so checkpoints from
    earlier runs load without translation.
    """

    def __init__(
        self,
        num_classes: int = 4,
        channels: int = 32,
        num_blocks: int = 5,
        dropout: float = 0.20,
        n_interval_features: int = 0,
    ) -> None:
        super().__init__()

        self.n_interval_features = int(n_interval_features)
        if self.n_interval_features < 0:
            raise ValueError("n_interval_features must be >= 0")

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
            nn.Linear(channels + self.n_interval_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intervals: torch.Tensor | None = None
        if self.n_interval_features:
            if x.shape[1] != 1 + self.n_interval_features:
                raise ValueError(
                    f"expected {1 + self.n_interval_features} input channels, got {x.shape[1]}"
                )
            # Each interval plane is constant along time, so one sample of it is
            # the whole feature.
            intervals = x[:, 1:, 0]
            x = x[:, :1, :]

        x = self.stem(x)
        x = self.blocks(x)
        x = self.global_pool(x)

        if intervals is not None:
            x = torch.cat([x.flatten(1), intervals], dim=1)

        return self.classifier(x)