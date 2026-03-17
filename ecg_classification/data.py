from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

from ecg_classification.augment import BeatAugmenter
from ecg_classification.constants import NUM_CLASSES, SAMPLE_LENGTH


@dataclass(slots=True)
class DatasetBundle:
    """Structured container for train, validation, and test arrays."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return pd.read_csv(path, header=None)


def _split_features_and_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    features = values[:, :-1]
    labels = values[:, -1].astype(np.int64)

    if features.shape[1] != SAMPLE_LENGTH:
        raise ValueError(
            f"Expected {SAMPLE_LENGTH} samples per beat, got {features.shape[1]}."
        )

    invalid_labels = labels[(labels < 0) | (labels >= NUM_CLASSES)]
    if invalid_labels.size > 0:
        observed = sorted(int(value) for value in np.unique(invalid_labels))
        raise ValueError(f"Labels must be in range [0, {NUM_CLASSES - 1}], found {observed}.")

    return features, labels


def load_mitbih_csv(train_csv: Path, test_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the official MIT-BIH train/test CSV files."""
    train_frame = _read_csv(train_csv)
    test_frame = _read_csv(test_csv)

    X_train, y_train = _split_features_and_labels(train_frame)
    X_test, y_test = _split_features_and_labels(test_frame)
    return X_train, y_train, X_test, y_test


def build_dataset_bundle(
    train_csv: Path,
    test_csv: Path,
    validation_size: float,
    random_state: int,
) -> DatasetBundle:
    """Create train, validation, and test partitions without leakage."""
    if not 0.0 < float(validation_size) < 1.0:
        raise ValueError(f"validation_size must be in (0, 1), got {validation_size}.")

    X_train_full, y_train_full, X_test, y_test = load_mitbih_csv(train_csv, test_csv)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    return DatasetBundle(
        X_train=X_train.astype(np.float32, copy=False),
        y_train=y_train.astype(np.int64, copy=False),
        X_valid=X_valid.astype(np.float32, copy=False),
        y_valid=y_valid.astype(np.int64, copy=False),
        X_test=X_test.astype(np.float32, copy=False),
        y_test=y_test.astype(np.int64, copy=False),
    )


def class_distribution(labels: np.ndarray) -> dict[int, int]:
    """Return a class-count dictionary."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {class_id: int(count) for class_id, count in enumerate(counts)}


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a weighted sampler to reduce class imbalance in training."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def materialize_augmented_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    augmenter: BeatAugmenter,
    augment_labels: Iterable[int],
    copies_per_sample: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a physically expanded dataset with augmented copies."""
    augment_label_set = set(int(label) for label in augment_labels)
    if copies_per_sample <= 0 or not augment_label_set:
        return (
            np.asarray(features, dtype=np.float32, copy=False),
            np.asarray(labels, dtype=np.int64, copy=False),
        )

    rng = np.random.default_rng(int(seed))
    base_features = np.asarray(features, dtype=np.float32)
    base_labels = np.asarray(labels, dtype=np.int64)

    augmented_features = [base_features]
    augmented_labels = [base_labels]

    eligible_indices = np.flatnonzero(np.isin(base_labels, list(augment_label_set)))
    for _ in range(int(copies_per_sample)):
        copied_features = np.empty((len(eligible_indices), base_features.shape[1]), dtype=np.float32)
        copied_labels = base_labels[eligible_indices].copy()
        for output_index, sample_index in enumerate(eligible_indices):
            copied_features[output_index] = augmenter(base_features[sample_index], rng)
        augmented_features.append(copied_features)
        augmented_labels.append(copied_labels)

    return (
        np.concatenate(augmented_features, axis=0).astype(np.float32, copy=False),
        np.concatenate(augmented_labels, axis=0).astype(np.int64, copy=False),
    )


class HeartbeatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for 1D heartbeat vectors."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmenter: BeatAugmenter | None = None,
        augment_labels: Iterable[int] | None = None,
        augment_probability: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"features and labels must have the same length, got {len(self.features)} and {len(self.labels)}."
            )

        self.augmenter = augmenter
        self.augment_labels = set(augment_labels or [])
        self.augment_probability = float(augment_probability)
        if not 0.0 <= self.augment_probability <= 1.0:
            raise ValueError(
                f"augment_probability must be in [0, 1], got {self.augment_probability}."
            )

        self.base_seed = int(seed)
        self._rng_by_worker_id: dict[int, np.random.Generator] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def _worker_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1

        if worker_id not in self._rng_by_worker_id:
            worker_seed = self.base_seed + (worker_id + 1) * 100_003
            self._rng_by_worker_id[worker_id] = np.random.default_rng(worker_seed)

        return self._rng_by_worker_id[worker_id]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.features[index]
        label = int(self.labels[index])
        rng = self._worker_rng()

        if (
            self.augmenter is not None
            and label in self.augment_labels
            and rng.random() < self.augment_probability
        ):
            signal = self.augmenter(signal, rng)

        signal = np.expand_dims(signal, axis=0).astype(np.float32, copy=False)
        return (
            torch.from_numpy(signal),
            torch.tensor(label, dtype=torch.long),
        )
