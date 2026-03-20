from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_classification.augment import BeatAugmenter
from ecg_classification.constants import CLASS_NAMES, CLASS_SYMBOLS, NUM_CLASSES
from ecg_classification.data import load_mitbih_csv, materialize_augmented_dataset
from ecg_classification.metrics import save_confusion_matrix, save_learning_curves
from ecg_classification.model import ResidualCNN
from ecg_classification.train import PROJECT_ROOT, TrainConfig
from ecg_classification.utils import ensure_directory


def _load_run_metrics(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Run directory is missing metrics.json: {metrics_path}. "
            "Run training first, or pass --run-dir to a completed run output directory."
        )
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_run_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Run directory is missing config.json: {config_path}. "
            "Run training first, or pass --run-dir to a completed run output directory."
        )
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _wrap_labels(labels: list[str], width: int = 16) -> list[str]:
    return [textwrap.fill(label, width=width) for label in labels]


def generate_eda_figures(
    train_csv: Path,
    test_csv: Path,
    output_dir: Path,
    augment_labels: tuple[int, ...],
    materialized_copies_per_sample: int,
) -> list[Path]:
    output_dir = ensure_directory(output_dir)
    X_train, y_train, X_test, y_test = load_mitbih_csv(train_csv, test_csv)

    generated_paths: list[Path] = []
    wrapped_class_names = _wrap_labels(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(NUM_CLASSES)
    train_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    test_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    width = 0.38
    ax.bar(positions - width / 2, train_counts, width=width, label="Train")
    ax.bar(positions + width / 2, test_counts, width=width, label="Test")
    ax.set_xticks(positions)
    ax.set_xticklabels(wrapped_class_names, rotation=18, ha="right")
    ax.set_ylabel("Number of beats")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    class_dist_path = output_dir / "eda_class_distribution.png"
    fig.savefig(class_dist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(class_dist_path)

    augmenter = BeatAugmenter()
    materialized_features, materialized_labels = materialize_augmented_dataset(
        X_train,
        y_train,
        augmenter=augmenter,
        augment_labels=augment_labels,
        copies_per_sample=materialized_copies_per_sample,
        seed=42,
    )
    original_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    materialized_counts = np.bincount(materialized_labels, minlength=NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.38
    ax.bar(positions - width / 2, original_counts, width=width, label="Before augmentation")
    ax.bar(positions + width / 2, materialized_counts, width=width, label="After materialized augmentation")
    ax.set_xticks(positions)
    ax.set_xticklabels(wrapped_class_names, rotation=18, ha="right")
    ax.set_ylabel("Number of beats")
    ax.set_title("Class counts before and after materialized augmentation")
    ax.legend()
    ax.text(
        0.01,
        0.98,
        f"Materialized setting: {materialized_copies_per_sample} synthetic copy/copies per eligible training beat.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    augmented_dist_path = output_dir / "eda_augmentation_class_distribution.png"
    fig.savefig(augmented_dist_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(augmented_dist_path)

    fig, axes = plt.subplots(NUM_CLASSES, 1, figsize=(11, 13), sharex=True)
    for class_index, axis in enumerate(np.atleast_1d(axes)):
        class_rows = np.flatnonzero(y_train == class_index)
        if class_rows.size == 0:
            axis.text(0.5, 0.5, "No sample available", ha="center", va="center")
            axis.set_title(CLASS_NAMES[class_index])
            axis.set_yticks([])
            continue

        signal = X_train[class_rows[0]]
        axis.plot(signal, color="#1f77b4", linewidth=1.2)
        axis.set_title(f"{CLASS_NAMES[class_index]} ({CLASS_SYMBOLS[class_index]})")
        axis.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Representative heartbeat waveforms", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    waveform_path = output_dir / "eda_representative_beats.png"
    fig.savefig(waveform_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(waveform_path)

    rng = np.random.default_rng(42)
    preview_labels = [label for label in augment_labels if np.any(y_train == label)]
    if preview_labels:
        fig, axes = plt.subplots(len(preview_labels), 2, figsize=(12, 3 * len(preview_labels)), sharex=True)
        axes = np.atleast_2d(axes)
        for row_index, class_index in enumerate(preview_labels):
            sample_index = int(np.flatnonzero(y_train == class_index)[0])
            original_signal = X_train[sample_index]
            augmented_signal = augmenter(original_signal, rng)

            left_axis = axes[row_index, 0]
            right_axis = axes[row_index, 1]
            left_axis.plot(original_signal, color="#2563eb", linewidth=1.2)
            right_axis.plot(augmented_signal, color="#dc2626", linewidth=1.2)
            left_axis.set_title(f"{CLASS_NAMES[class_index]}: original", fontsize=10)
            right_axis.set_title(f"{CLASS_NAMES[class_index]}: augmented", fontsize=10)
            left_axis.set_ylabel("Amplitude")

        axes[-1, 0].set_xlabel("Sample index")
        axes[-1, 1].set_xlabel("Sample index")
        fig.suptitle("Augmentation examples by class", y=0.995, fontsize=14)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
        preview_path = output_dir / "eda_augmentation_waveform_examples.png"
        fig.savefig(preview_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        generated_paths.append(preview_path)

    return generated_paths


def generate_method_figure(output_dir: Path) -> list[Path]:
    output_dir = ensure_directory(output_dir)
    generated_paths: list[Path] = []

    model = ResidualCNN(num_classes=NUM_CLASSES)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axis("off")

    blocks = [
        ("Input", "Beat vector\n1 x 187"),
        ("Stem", "Conv1d\nBatchNorm1d\nReLU"),
        ("Residual stack", "5 x ResidualBlock1D"),
        ("Global pooling", "AdaptiveAvgPool1d(1)"),
        ("Classifier", "Linear(32->64)\nReLU\nDropout\nLinear(64->5)"),
        ("Output", f"{NUM_CLASSES} class logits"),
    ]

    start_x = 0.03
    width = 0.135
    gap = 0.025
    for index, (title, subtitle) in enumerate(blocks):
        x = start_x + index * (width + gap)
        rect = plt.Rectangle((x, 0.62), width, 0.18, facecolor="#e8f1fb", edgecolor="#2a5c8a", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.74, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + width / 2, 0.67, subtitle, ha="center", va="center", fontsize=8.6)
        if index < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.7, 0.71),
                xytext=(x + width, 0.71),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#444444"},
            )

    text_box = plt.Rectangle((0.03, 0.14), 0.94, 0.33, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.2)
    ax.add_patch(text_box)
    pipeline_text = (
        "Training pipeline\n"
        "1. Load MIT-BIH train/test CSV files.\n"
        "2. Split the training partition into train/validation.\n"
        "3. Choose augmentation mode: none, on-the-fly, or materialized static augmentation.\n"
        "4. Optionally apply weighted sampling to reduce class imbalance during training.\n"
        "5. Train ResidualCNN with AdamW, ReduceLROnPlateau, and early stopping.\n"
        "6. Evaluate on the held-out test set and export metrics plus figure assets."
    )
    ax.text(0.05, 0.43, pipeline_text, fontsize=10, va="top", ha="left", linespacing=1.45)
    ax.text(0.03, 0.07, f"Trainable parameters: {parameter_count:,}", fontsize=10)
    ax.set_title("Method overview: model architecture and training workflow", fontsize=14, pad=12)

    method_path = output_dir / "method_model_architecture.png"
    fig.savefig(method_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(method_path)

    return generated_paths


def generate_result_figures(run_dir: Path, output_dir: Path) -> list[Path]:
    output_dir = ensure_directory(output_dir)
    generated_paths: list[Path] = []

    metrics = _load_run_metrics(run_dir)
    config = _load_run_config(run_dir)
    history = pd.read_csv(run_dir / "history.csv")

    confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    confusion_path = output_dir / "result_confusion_matrix.png"
    save_confusion_matrix(confusion_matrix, confusion_path, normalize=True)
    generated_paths.append(confusion_path)

    learning_curve_path = output_dir / "result_learning_curves.png"
    save_learning_curves(history, learning_curve_path, configured_max_epoch=int(config.get("epochs", int(history["epoch"].max()))))
    generated_paths.append(learning_curve_path)

    summary_metrics = {
        "Accuracy": float(metrics["accuracy"]),
        "Macro F1": float(metrics["macro_f1"]),
        "LRAP": float(metrics["label_ranking_average_precision"]),
        "1 - Rank loss": 1.0 - float(metrics["label_ranking_loss"]),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(summary_metrics)
    values = list(summary_metrics.values())
    bars = ax.bar(names, values, color=["#3b82f6", "#2563eb", "#0f766e", "#0891b2"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Result summary metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    summary_path = output_dir / "result_metric_summary.png"
    fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(summary_path)

    return generated_paths


def generate_augmentation_comparison_figure(run_specs: list[tuple[str, Path]], output_dir: Path) -> list[Path]:
    output_dir = ensure_directory(output_dir)
    generated_paths: list[Path] = []
    if len(run_specs) < 2:
        return generated_paths

    metric_names = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("label_ranking_average_precision", "LRAP"),
    ]

    run_labels = [label for label, _ in run_specs]
    metric_values = {title: [] for _, title in metric_names}
    for _, run_dir in run_specs:
        metrics = _load_run_metrics(run_dir)
        for metric_key, metric_title in metric_names:
            metric_values[metric_title].append(float(metrics[metric_key]))

    fig, axes = plt.subplots(1, len(metric_names), figsize=(13, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    colors = ["#2563eb", "#059669", "#dc2626"]
    for axis, (metric_key, metric_title) in zip(axes, metric_names):
        values = metric_values[metric_title]
        bars = axis.bar(run_labels, values, color=colors[: len(run_labels)])
        axis.set_title(metric_title)
        axis.set_ylim(0.0, 1.05)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Score")
    fig.suptitle("Augmentation strategy comparison", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = output_dir / "result_augmentation_comparison.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    generated_paths.append(output_path)
    return generated_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA, method, and result figures for the ECG project.")
    parser.add_argument("--train-csv", type=Path, default=PROJECT_ROOT / "data" / "mitbih" / "mitbih_train.csv")
    parser.add_argument("--test-csv", type=Path, default=PROJECT_ROOT / "data" / "mitbih" / "mitbih_test.csv")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "baseline_run")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "assets")
    parser.add_argument(
        "--augment-labels",
        type=int,
        nargs="*",
        default=list(TrainConfig().augment_labels),
        help="Class labels to include in augmentation preview figures.",
    )
    parser.add_argument(
        "--materialized-copies-per-sample",
        type=int,
        default=TrainConfig().materialized_copies_per_sample,
        help="Number of synthetic copies per eligible sample for augmentation EDA figures.",
    )
    parser.add_argument(
        "--compare-run",
        nargs=2,
        action="append",
        metavar=("LABEL", "RUN_DIR"),
        help="Add a labeled run directory for augmentation comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eda_dir = ensure_directory(args.output_dir / "eda")
    method_dir = ensure_directory(args.output_dir / "method")
    result_dir = ensure_directory(args.output_dir / "result")

    generated_paths = []
    generated_paths.extend(
        generate_eda_figures(
            args.train_csv,
            args.test_csv,
            eda_dir,
            tuple(args.augment_labels),
            args.materialized_copies_per_sample,
        )
    )
    generated_paths.extend(generate_method_figure(method_dir))
    generated_paths.extend(generate_result_figures(args.run_dir, result_dir))
    if args.compare_run:
        comparison_specs = [(label, Path(run_dir)) for label, run_dir in args.compare_run]
        generated_paths.extend(generate_augmentation_comparison_figure(comparison_specs, result_dir))

    print("Generated figures:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
