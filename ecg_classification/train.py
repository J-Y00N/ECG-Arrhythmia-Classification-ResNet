from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ecg_classification.augment import BeatAugmenter
from ecg_classification.constants import NUM_CLASSES
from ecg_classification.data import (
    HeartbeatDataset,
    build_dataset_bundle,
    class_distribution,
    materialize_augmented_dataset,
    make_weighted_sampler,
)
from ecg_classification.metrics import (
    evaluate_model,
    save_confusion_matrix,
    save_learning_curves,
    save_metrics_bundle,
)
from ecg_classification.model import ResidualCNN
from ecg_classification.utils import NumpyJSONEncoder, configure_torch_runtime, default_device, ensure_directory, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class TrainConfig:
    train_csv: Path = PROJECT_ROOT / "data" / "mitbih" / "mitbih_train.csv"
    test_csv: Path = PROJECT_ROOT / "data" / "mitbih" / "mitbih_test.csv"
    output_dir: Path = PROJECT_ROOT / "outputs" / "baseline_run"
    validation_size: float = 0.10
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    use_weighted_sampler: bool = True
    augment_probability: float = 0.60
    augment_labels: tuple[int, ...] = (1, 2, 3, 4)
    augmentation_mode: str = "on_the_fly"
    materialized_copies_per_sample: int = 1
    label_smoothing: float = 0.05
    show_progress: bool = False


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
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, dict[int, int]]]:
    """Create train, validation, and test dataloaders."""
    bundle = build_dataset_bundle(
        train_csv=config.train_csv,
        test_csv=config.test_csv,
        validation_size=config.validation_size,
        random_state=config.seed,
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
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    distributions = {
        "train": class_distribution(train_labels),
        "valid": class_distribution(bundle.y_valid),
        "test": class_distribution(bundle.y_test),
    }

    return train_loader, valid_loader, test_loader, distributions


def run_training(config: TrainConfig) -> dict[str, Any]:
    """Train the model and save outputs."""
    set_seed(config.seed)
    device = default_device()
    configure_torch_runtime(device)
    output_dir = ensure_directory(config.output_dir)

    train_loader, valid_loader, test_loader, distributions = build_dataloaders(config, device)

    model = ResidualCNN(num_classes=NUM_CLASSES).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
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

        scheduler.step(valid_metrics["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]
        history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "valid_loss": float(valid_metrics["loss"]),
                "valid_accuracy": float(valid_metrics["accuracy"]),
                "valid_macro_f1": float(valid_metrics["macro_f1"]),
                "learning_rate": float(current_lr),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_acc={valid_metrics['accuracy']:.4f} | "
            f"valid_macro_f1={valid_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if float(valid_metrics["macro_f1"]) > best_valid_f1:
            best_valid_f1 = float(valid_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if early_stopping.step(float(valid_metrics["macro_f1"])):
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

    torch.save(best_state, output_dir / "best_model.pt")

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "history.csv", index=False)

    save_metrics_bundle(test_metrics, output_dir)
    save_confusion_matrix(test_metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_learning_curves(history_frame, output_dir / "learning_curves.png", configured_max_epoch=config.epochs)

    config_payload = asdict(config)
    config_payload["train_csv"] = str(config.train_csv)
    config_payload["test_csv"] = str(config.test_csv)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["device"] = str(device)
    config_payload["best_epoch"] = best_epoch
    config_payload["best_valid_macro_f1"] = best_valid_f1
    config_payload["class_distributions"] = distributions

    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config_payload, file, indent=2, cls=NumpyJSONEncoder)

    result = {
        "output_dir": output_dir,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_valid_f1,
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }
    return result


def parse_args() -> TrainConfig:
    defaults = TrainConfig()

    parser = argparse.ArgumentParser(description="Train the refactored ECG classifier.")
    parser.add_argument("--train-csv", type=Path, default=defaults.train_csv)
    parser.add_argument("--test-csv", type=Path, default=defaults.test_csv)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
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
        "--disable-weighted-sampler",
        action="store_true",
        help="Disable the weighted training sampler.",
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

    augmentation_mode = "none" if args.disable_augmentation else args.augmentation_mode

    return TrainConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_weighted_sampler=not args.disable_weighted_sampler,
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
        f"best_epoch={result['best_epoch']} | "
        f"best_valid_macro_f1={result['best_valid_macro_f1']:.4f} | "
        f"test_accuracy={result['test_accuracy']:.4f} | "
        f"test_macro_f1={result['test_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
