from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample

from ecg_classification.constants import SAMPLE_LENGTH


def _resize_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a 1D signal to a new length."""
    resized = resample(signal, target_length).astype(np.float32)
    return resized


def time_stretch(
    signal: np.ndarray,
    rng: np.random.Generator,
    sample_length: int = SAMPLE_LENGTH,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    """Randomly stretch or compress the signal in time."""
    length = int(sample_length * rng.uniform(min_scale, max_scale))
    length = max(8, length)

    stretched = _resize_signal(signal, length)
    if length == sample_length:
        return stretched

    output = np.zeros(sample_length, dtype=np.float32)
    if length < sample_length:
        output[:length] = stretched
        return output

    return stretched[:sample_length]


def amplitude_scale(
    signal: np.ndarray,
    rng: np.random.Generator,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    """Apply a simple amplitude scaling factor."""
    scale = rng.uniform(min_scale, max_scale)
    return (signal * scale).astype(np.float32)


def add_gaussian_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Inject a small amount of Gaussian noise."""
    noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


@dataclass(slots=True)
class BeatAugmenter:
    """On-the-fly augmentation for heartbeat vectors."""

    sample_length: int = SAMPLE_LENGTH
    stretch_probability: float = 0.50
    amplitude_probability: float = 0.50
    noise_probability: float = 0.30
    clip_min: float = 0.0
    clip_max: float = 1.0

    def __call__(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        augmented = signal.astype(np.float32, copy=True)

        if rng.random() < self.stretch_probability:
            augmented = time_stretch(
                augmented,
                rng=rng,
                sample_length=self.sample_length,
            )

        if rng.random() < self.amplitude_probability:
            augmented = amplitude_scale(augmented, rng=rng)

        if rng.random() < self.noise_probability:
            augmented = add_gaussian_noise(augmented, rng=rng)

        return np.clip(augmented, self.clip_min, self.clip_max).astype(np.float32)
