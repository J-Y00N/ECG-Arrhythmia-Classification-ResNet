from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ecg_classification.augment import BeatAugmenter
from ecg_classification.constants import CLASS_SYMBOLS, NUM_CLASSES, OBSERVATION_SYMBOL
from ecg_classification.data import (
    DatasetBundle,
    HeartbeatDataset,
    build_dataset_bundle,
    class_distribution,
    class_weights,
    materialize_augmented_dataset,
    make_weighted_sampler,
    record_class_distribution,
    scored_classes,
)
from ecg_classification.metrics import (
    evaluate_model,
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_bundle,
)
from ecg_classification.model import ResidualCNN
from ecg_classification.predictions import collect_predictions, save_predictions
from ecg_classification.utils import NumpyJSONEncoder, configure_torch_runtime, default_device, ensure_directory, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Classes used for model selection. Fixed here rather than derived per split.
#:
#: The fusion class cannot support model selection under the inter-patient
#: protocol. Ninety percent of DS1's fusion beats come from record 208, so a
#: validation split either holds that record -- leaving training with about
#: forty fusion beats -- or it does not, leaving validation with almost none.
#: The two requirements are incompatible, and no choice of validation size
#: resolves it.
#:
#: Deriving the set per split instead makes it vary by seed: at a validation
#: size of 0.10 one seed scored on N and S while another scored on N and V, so
#: the seeds were selecting their models against different objectives and
#: averaging across them meant nothing. Fixing the set costs a little
#: information about fusion during training and buys a comparison that holds.
#:
#: Selection is restricted; reporting is not. Test metrics cover all four
#: classes, and DS2 carries enough fusion beats to score them.
SELECTION_CLASSES: tuple[int, ...] = (0, 1, 2)


@dataclass(slots=True)
class TrainConfig:
    protocol: str = "inter"
    output_dir: Path | None = None
    data_dir: Path = PROJECT_ROOT / "data"
    strict_disjoint: bool = False
    validation_size: float = 0.20
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    use_class_weighted_loss: bool = True
    weight_power: float = 0.5
    use_weighted_sampler: bool = False
    augment_probability: float = 0.60
    augment_labels: tuple[int, ...] = (1, 2, 3)
    augmentation_mode: str = "on_the_fly"
    materialized_copies_per_sample: int = 1
    label_smoothing: float = 0.05
    show_progress: bool = False

    def run_name(self) -> str:
        """Directory name encoding every factor that varies across the matrix.

        Runs differ only in protocol, augmentation mode, rebalancing scheme and
        seed, so naming by those four keeps twenty output directories readable
        and makes an accidental overwrite obvious rather than silent.
        """
        if self.use_class_weighted_loss:
            rebalancing = f"lossweight{self.weight_power:g}"
        elif self.use_weighted_sampler:
            rebalancing = "sampler"
        else:
            rebalancing = "noreweight"
        strict = "-strict" if self.strict_disjoint else ""
        return f"{self.protocol}{strict}-{self.augmentation_mode}-{rebalancing}-seed{self.seed}"

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return PROJECT_ROOT / "outputs" / self.run_name()


class EarlyStopping:
    """Simple early stopping on a maximized validation metric."""

    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best_score = float("-inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        score = float(score)
        if score > self.best_score:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def macro_f1_from_confusion(matrix: np.ndarray, class_indices: list[int]) -> float:
    """Macro F1 restricted to the given classes, computed from a confusion matrix.

    Model selection needs this because the validation half of the inter-patient
    protocol is only two recordings, and minority classes in this database sit
    in a handful of recordings. A validation split can therefore contain no
    fusion beats at all, in which case an unrestricted macro average scores an
    absent class as zero and drags the metric down by a quarter every epoch.
    The offset is constant, so the ranking of epochs survives, but the number
    reported alongside it would be meaningless.

    Restricting the average to classes that are actually present keeps the
    metric about the model. Which classes those were is written into
    ``config.json``, so nothing is hidden by the restriction.
    """
    confusion = np.asarray(matrix, dtype=np.float64)
    scores: list[float] = []
    for index in class_indices:
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(scores)) if scores else 0.0


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    show_progress: bool,
) -> dict[str, float]:
    """Run one training epoch."""
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress = tqdm(dataloader, desc="Train", leave=False, disable=not show_progress)
    for inputs, labels in progress:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        running_correct += int((predictions == labels).sum().item())
        total_samples += labels.size(0)
        running_loss += float(loss.item()) * labels.size(0)

        average_loss = running_loss / max(total_samples, 1)
        progress.set_postfix(loss=f"{average_loss:.4f}")

    return {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": running_correct / max(total_samples, 1),
    }


def build_dataloaders(
    config: TrainConfig,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DatasetBundle, np.ndarray, dict[str, Any]]:
    """Create dataloaders for training, validation, test and the observation set."""
    bundle = build_dataset_bundle(
        protocol=config.protocol,
        seed=config.seed,
        validation_size=config.validation_size,
        data_dir=config.data_dir,
        strict_disjoint=config.strict_disjoint,
    )

    augmenter = BeatAugmenter()
    train_features = bundle.X_train
    train_labels = bundle.y_train

    dataset_augmenter: BeatAugmenter | None = None
    dataset_augment_probability = 0.0
    if config.augmentation_mode == "materialized":
        train_features, train_labels = materialize_augmented_dataset(
            train_features,
            train_labels,
            augmenter=augmenter,
            augment_labels=config.augment_labels,
            copies_per_sample=config.materialized_copies_per_sample,
            seed=config.seed,
        )
    elif config.augmentation_mode == "on_the_fly":
        dataset_augmenter = augmenter
        dataset_augment_probability = config.augment_probability

    train_dataset = HeartbeatDataset(
        train_features,
        train_labels,
        augmenter=dataset_augmenter,
        augment_labels=config.augment_labels,
        augment_probability=dataset_augment_probability,
        seed=config.seed,
    )
    valid_dataset = HeartbeatDataset(bundle.X_valid, bundle.y_valid, seed=config.seed)
    test_dataset = HeartbeatDataset(bundle.X_test, bundle.y_test, seed=config.seed)

    # The observation set carries no model label; zeros are placeholders and the
    # true-label column is overwritten before the table is written out.
    observe_dataset = HeartbeatDataset(
        bundle.X_observe,
        np.zeros(len(bundle.X_observe), dtype=np.int64),
        seed=config.seed,
    )

    sampler = make_weighted_sampler(train_labels) if config.use_weighted_sampler else None

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        **loader_kwargs,
    )
    # Evaluation loaders must not shuffle or drop a partial batch: the
    # prediction table pairs their output with record identifiers by position.
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    observe_loader = DataLoader(observe_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    distributions: dict[str, Any] = {
        "train": class_distribution(train_labels),
        "valid": class_distribution(bundle.y_valid),
        "test": class_distribution(bundle.y_test),
        "train_by_record": record_class_distribution(bundle.y_train, bundle.r_train),
        "valid_by_record": record_class_distribution(bundle.y_valid, bundle.r_valid),
    }

    return train_loader, valid_loader, test_loader, observe_loader, bundle, train_labels, distributions


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Train the model and save outputs."""
    set_seed(config.seed)
    device = default_device()
    configure_torch_runtime(device)
    output_dir = ensure_directory(config.resolved_output_dir())

    (
        train_loader,
        valid_loader,
        test_loader,
        observe_loader,
        bundle,
        train_labels,
        distributions,
    ) = build_dataloaders(config, device)

    # Weights are computed from the labels the network actually sees, which
    # differ from the bundle's under materialized augmentation.
    weight = (
        class_weights(train_labels, power=config.weight_power).to(device)
        if config.use_class_weighted_loss
        else None
    )
    valid_scored = list(SELECTION_CLASSES)
    valid_supported = scored_classes(bundle.y_valid)

    missing = [c for c in valid_scored if c not in valid_supported]
    if missing:
        print(
            f"Warning: selection classes {missing} have little or no support in this "
            f"validation split (supported: {valid_supported}). Selection proceeds on "
            f"the fixed set so that seeds remain comparable."
        )

    model = ResidualCNN(num_classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    early_stopping = EarlyStopping(patience=config.patience)

    history_rows: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_valid_f1 = float("-inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            show_progress=config.show_progress,
        )
        valid_metrics = evaluate_model(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        selection_f1 = macro_f1_from_confusion(valid_metrics["confusion_matrix"], valid_scored)
        scheduler.step(selection_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "valid_loss": float(valid_metrics["loss"]),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
                "valid_macro_f1_scored": float(selection_f1),
                "learning_rate": float(current_lr),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f} | "
            f"valid_macro_f1={selection_f1:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if selection_f1 > best_valid_f1:
            best_valid_f1 = selection_f1
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if early_stopping.step(selection_f1):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    # One row per beat, with the recording it came from. Every patient-level
    # analysis downstream reads this file rather than re-running inference.
    prediction_frame = collect_predictions(
        model=model,
        dataloader=test_loader,
        record_ids=bundle.r_test,
        beat_indices=bundle.split.test_index,
        device=device,
    )
    save_predictions(prediction_frame, output_dir)

    # The unclassifiable beats, which the model never saw. Not scored: fifteen
    # beats is nowhere near enough. Kept because how a classifier trained only
    # on clean morphology responds to pure artefact is a calibration question
    # worth looking at, and confident output on a detached-electrode waveform
    # would be a clinically dangerous failure mode.
    if len(bundle.X_observe):
        observation_frame = collect_predictions(
            model=model,
            dataloader=observe_loader,
            record_ids=bundle.r_observe,
            beat_indices=np.arange(len(bundle.r_observe)),
            device=device,
        )
        observation_frame["y_true"] = -1
        observation_frame.to_csv(output_dir / "observations.csv", index=False, float_format="%.6f")

    torch.save(best_state, output_dir / "best_model.pt")

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "history.csv", index=False)

    save_metrics_bundle(test_metrics, output_dir)
    save_confusion_matrix(test_metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_learning_curves(history_frame, output_dir / "learning_curves.png", configured_max_epoch=config.epochs)

    test_scored = scored_classes(bundle.y_test)
    test_macro_f1_scored = macro_f1_from_confusion(test_metrics["confusion_matrix"], test_scored)

    config_payload = asdict(config)
    config_payload["output_dir"] = str(output_dir)
    config_payload["data_dir"] = str(config.data_dir)
    config_payload["run_name"] = config.run_name()
    config_payload["device"] = str(device)
    config_payload["best_epoch"] = best_epoch
    config_payload["best_valid_macro_f1"] = best_valid_f1
    config_payload["class_symbols"] = list(CLASS_SYMBOLS)
    config_payload["observation_symbol"] = OBSERVATION_SYMBOL
    config_payload["n_observation_beats"] = int(len(bundle.X_observe))
    config_payload["class_distributions"] = distributions
    config_payload["class_weights"] = None if weight is None else weight.cpu().tolist()
    config_payload["selection_classes"] = valid_scored
    config_payload["valid_supported_classes"] = valid_supported
    config_payload["test_scored_classes"] = test_scored
    # The split description is what makes the "only the protocol changed" claim
    # checkable: diffing two runs' config.json should show one differing field.
    config_payload["split"] = bundle.split.describe()

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2, cls=NumpyJSONEncoder)

    return {
        "output_dir": output_dir,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_f1,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_macro_f1_scored": float(test_macro_f1_scored),
    }


def parse_args() -> TrainConfig:
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Train the refactored ECG classifier.")
    parser.add_argument(
        "--protocol",
        choices=["intra", "inter"],
        default=defaults.protocol,
        help="inter assigns whole recordings following de Chazal; intra splits at the beat level.",
    )
    parser.add_argument(
        "--strict-disjoint",
        action="store_true",
        help="Drop record 202, whose subject also appears in DS1 as record 201.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/<protocol>-<augmentation>-<rebalancing>-seed<n>.",
    )
    parser.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    parser.add_argument("--validation-size", type=float, default=defaults.validation_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--augment-probability", type=float, default=defaults.augment_probability)
    parser.add_argument(
        "--augmentation-mode",
        choices=["none", "on_the_fly", "materialized"],
        default=defaults.augmentation_mode,
        help="Choose how augmentation is applied during training.",
    )
    parser.add_argument(
        "--materialized-copies-per-sample",
        type=int,
        default=defaults.materialized_copies_per_sample,
        help="Number of synthetic copies per eligible sample when using materialized augmentation.",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Alias for --augmentation-mode none.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show per-batch tqdm progress bars.",
    )
    parser.add_argument(
        "--weight-power",
        type=float,
        default=defaults.weight_power,
        help=(
            "Exponent on the inverse-frequency class weights. 0 is uniform, 1 is plain "
            "inverse frequency. On this data 1 collapses training onto the rare classes, "
            "so how much rebalancing is tolerable is itself worth measuring."
        ),
    )
    parser.add_argument(
        "--disable-class-weighted-loss",
        action="store_true",
        help="Train with an unweighted loss.",
    )
    parser.add_argument(
        "--enable-weighted-sampler",
        action="store_true",
        help=(
            "Rebalance by oversampling instead of by loss weight. Kept for comparison "
            "with the previous pipeline: minority classes here sit in a few recordings, "
            "so oversampling replicates one patient's morphology rather than adding any."
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.validation_size < 1.0:
        parser.error("--validation-size must be in (0, 1).")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer.")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer.")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be > 0.")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be >= 0.")
    if args.patience <= 0:
        parser.error("--patience must be a positive integer.")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")
    if not 0.0 <= args.augment_probability <= 1.0:
        parser.error("--augment-probability must be in [0, 1].")
    if args.materialized_copies_per_sample <= 0:
        parser.error("--materialized-copies-per-sample must be a positive integer.")
    if args.weight_power < 0.0:
        parser.error("--weight-power must be >= 0.")
    if args.enable_weighted_sampler and not args.disable_class_weighted_loss:
        parser.error(
            "--enable-weighted-sampler rebalances the data and --class-weighted-loss "
            "rebalances the loss. Applying both compounds the correction; pass "
            "--disable-class-weighted-loss as well if the sampler is what you want."
        )

    augmentation_mode = "none" if args.disable_augmentation else args.augmentation_mode

    return TrainConfig(
        protocol=args.protocol,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        strict_disjoint=args.strict_disjoint,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_class_weighted_loss=not args.disable_class_weighted_loss,
        weight_power=args.weight_power,
        use_weighted_sampler=args.enable_weighted_sampler,
        augment_probability=0.0 if augmentation_mode == "none" else args.augment_probability,
        augmentation_mode=augmentation_mode,
        materialized_copies_per_sample=args.materialized_copies_per_sample,
        show_progress=args.show_progress or sys.stderr.isatty(),
    )


def main() -> None:
    config = parse_args()
    result = run_training(config)
    print(
        "Training complete | "
        f"run={config.run_name()} | "
        f"best_epoch={result['best_epoch']} | "
        f"best_valid_macro_f1={result['best_valid_macro_f1']:.4f} | "
        f"test_accuracy={result['test_accuracy']:.4f} | "
        f"test_macro_f1={result['test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()