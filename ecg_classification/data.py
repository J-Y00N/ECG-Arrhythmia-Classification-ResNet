from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

from ecg_classification import mitdb
from ecg_classification.augment import BeatAugmenter
from ecg_classification.constants import NUM_CLASSES, OBSERVATION_LABEL, OBSERVATION_SYMBOL, SAMPLE_LENGTH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(slots=True)
class DatasetBundle:
    """Structured container for train, validation, and test arrays.

    Every split carries its record identifiers alongside the signals. Those
    identifiers are what make patient-level analysis possible after the fact:
    the cluster bootstrap, the per-record decomposition and the effective
    patient count all need to know which recording a beat came from, and the
    previous CSV source discarded that information.

    Beats carrying the observation label are held out of all three splits and
    returned separately in ``X_observe``. They are never trained on, never
    validated on and never scored; they exist so that the behaviour of a
    classifier meeting pure artefact can be looked at, which is a calibration
    question rather than a classification one.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    r_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    r_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    r_test: np.ndarray
    X_observe: np.ndarray
    r_observe: np.ndarray
    split: mitdb.Split


def _validate_arrays(features: np.ndarray, labels: np.ndarray) -> None:
    """Guard the assumptions the model and augmenter rely on."""
    if features.shape[1] != SAMPLE_LENGTH:
        raise ValueError(
            f"Expected {SAMPLE_LENGTH} samples per beat, got {features.shape[1]}."
        )

    invalid_labels = labels[(labels < 0) | (labels >= NUM_CLASSES)]
    if invalid_labels.size > 0:
        observed = sorted(int(value) for value in np.unique(invalid_labels))
        raise ValueError(f"Labels must be in range [0, {NUM_CLASSES - 1}], found {observed}.")


def build_dataset_bundle(
    protocol: str = "inter",
    seed: int = 42,
    validation_size: float = 0.10,
    data_dir: Path = DEFAULT_DATA_DIR,
    strict_disjoint: bool = False,
) -> DatasetBundle:
    """Create train, validation, and test partitions from the raw-record cache.

    Replaces the CSV reader. All partitioning happens in :mod:`mitdb`, which
    owns the definition of the two protocols; this function only slices the
    cached arrays and checks that their shape still matches what the network
    expects.

    Under ``inter`` the validation half is carved out of DS1 by record, not by
    beat. Holding out beats from records the model also trains on would put the
    leakage back into model selection, which is the one place it is easiest to
    overlook.
    """
    if not 0.0 < float(validation_size) < 1.0:
        raise ValueError(f"validation_size must be in (0, 1), got {validation_size}.")

    cache = mitdb.load_cache(data_dir)
    features, labels, record_ids = cache["x"], cache["y"], cache["record_id"]

    # Split the observation class off before anything else. It is filtered here
    # rather than at cache-build time so the cache stays a faithful record of
    # what the database contains; deciding what to model is a modelling-layer
    # concern, and reversing this choice must not cost a 70 MB rebuild.
    observed = labels == OBSERVATION_LABEL
    modelled = ~observed
    X_observe = features[observed].astype(np.float32, copy=False)
    r_observe = record_ids[observed]

    features, labels, record_ids = features[modelled], labels[modelled], record_ids[modelled]
    _validate_arrays(features, labels)

    split = mitdb.make_split(
        record_ids=record_ids,
        labels=labels,
        protocol=protocol,
        seed=seed,
        validation_size=validation_size,
        strict_disjoint=strict_disjoint,
    )

    def take(index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            features[index].astype(np.float32, copy=False),
            labels[index].astype(np.int64, copy=False),
            record_ids[index],
        )

    return DatasetBundle(
        *take(split.train_index),
        *take(split.valid_index),
        *take(split.test_index),
        X_observe=X_observe,
        r_observe=r_observe,
        split=split,
    )


def class_distribution(labels: np.ndarray) -> dict[int, int]:
    """Return a class-count dictionary."""
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return {class_id: int(count) for class_id, count in enumerate(counts)}


def record_class_distribution(
    labels: np.ndarray, record_ids: np.ndarray
) -> dict[str, dict[int, int]]:
    """Per-record class counts for one split.

    Written into ``config.json`` so that a run's artefacts record not only how
    many beats of each class it saw but how many recordings those beats came
    from. In this database the two are very different numbers.
    """
    distribution: dict[str, dict[int, int]] = {}
    for record in np.unique(record_ids):
        mask = record_ids == record
        distribution[str(record)] = class_distribution(labels[mask])
    return distribution


def class_weights(labels: np.ndarray, min_support: int = 50) -> torch.Tensor:
    """Inverse-frequency class weights for a weighted loss.

    Preferred to oversampling in the inter-patient setting. Minority classes
    here are concentrated in a handful of recordings -- record 208 supplies
    about 90% of DS1's fusion beats -- so replicating them does not add patient
    diversity, it just shows the network one person's morphology many more
    times. Reweighting the loss achieves the same rebalancing without
    duplicating anything.

    Classes with fewer than ``min_support`` training beats receive zero weight,
    which removes them from the loss entirely. With four classes this floor
    should never fire -- the smallest, fusion, holds a few hundred training
    beats -- and it is kept only as a guard against a validation record split
    that strips a class almost bare. It exists because inverse frequency
    equalises the *total* contribution of every class, so a class reduced to a
    handful of beats would otherwise be weighted into the hundreds and
    destabilise training on its own.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    eligible = counts >= max(1, int(min_support))
    if not eligible.any():
        raise ValueError(
            f"no class reaches min_support={min_support}; counts were {counts.astype(int).tolist()}"
        )

    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    weights[eligible] = counts[eligible].sum() / (eligible.sum() * counts[eligible])
    return torch.as_tensor(weights, dtype=torch.float32)


def scored_classes(labels: np.ndarray, min_support: int = 50) -> list[int]:
    """Class indices with enough support to carry an interpretable metric.

    Macro-averaged scores treat every class equally, so a class holding seven
    artefact beats would move the headline number as much as one holding forty
    thousand normal beats. Restricting the average to classes above the support
    floor keeps the metric about the model rather than about how a handful of
    unclassifiable beats happened to fall.

    Report the restricted average as the headline figure and the full average
    alongside it, so nothing is hidden.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return [index for index, count in enumerate(counts) if count >= max(1, int(min_support))]


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a weighted sampler to reduce class imbalance in training.

    Retained for comparison with the previous pipeline. See
    :func:`class_weights` for why loss reweighting is the better default under
    the inter-patient protocol.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    class_weight = 1.0 / counts
    sample_weights = class_weight[labels]

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
    """PyTorch dataset for 1D heartbeat vectors.

    Deliberately still returns a ``(signal, label)`` pair. Record identifiers
    travel next to the dataset in :class:`DatasetBundle` rather than through
    the batch, so the training loop needs no change and evaluation can align
    identifiers positionally against an unshuffled loader.
    """

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