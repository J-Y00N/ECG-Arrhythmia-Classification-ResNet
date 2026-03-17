from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    coverage_error,
    f1_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
)
from sklearn.preprocessing import label_binarize

from ecg_classification.constants import CLASS_NAMES
from ecg_classification.utils import NumpyJSONEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate a model and collect classification metrics."""
    model.eval()

    running_loss = 0.0
    total_samples = 0
    probabilities: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probabilities.append(probs)
        labels_list.append(labels.cpu().numpy())

    if not labels_list or not probabilities:
        raise ValueError("Dataloader produced no samples during evaluation.")

    labels = list(range(len(CLASS_NAMES)))
    y_true = np.concatenate(labels_list, axis=0)
    y_prob = np.concatenate(probabilities, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true_one_hot = label_binarize(y_true, classes=labels)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    metrics = {
        "loss": running_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "label_ranking_average_precision": float(
            label_ranking_average_precision_score(y_true_one_hot, y_prob)
        ),
        "label_ranking_loss": float(label_ranking_loss(y_true_one_hot, y_prob)),
        "coverage_error": float(coverage_error(y_true_one_hot, y_prob)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def save_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Save a confusion matrix figure."""
    matrix = cm.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        matrix = matrix / row_sums

    wrapped_labels = [textwrap.fill(name, width=16) for name in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax)

    tick_positions = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(wrapped_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(wrapped_labels, fontsize=9)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    fmt = ".2f" if normalize else "d"
    threshold = matrix.max() / 2.0 if matrix.size else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(
    history: pd.DataFrame,
    output_path: Path,
    configured_max_epoch: int | None = None,
) -> None:
    """Save train/validation learning curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["valid_loss"], label="Validation loss")
    ax.plot(history["epoch"], history["valid_macro_f1"], label="Validation macro F1")
    max_epoch = int(history["epoch"].max())
    if configured_max_epoch is None:
        configured_max_epoch = max_epoch
    if max_epoch < configured_max_epoch:
        ax.axvline(
            configured_max_epoch,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label=f"Configured max epoch ({configured_max_epoch})",
        )
    ax.set_xlim(1, configured_max_epoch)
    ax.set_xlabel("Epoch")
    ax.set_title("Training History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bundle(metrics: dict[str, object], output_dir: Path) -> None:
    """Write metrics to text and JSON files."""
    metrics_json = {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "label_ranking_average_precision": metrics["label_ranking_average_precision"],
        "label_ranking_loss": metrics["label_ranking_loss"],
        "coverage_error": metrics["coverage_error"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report_dict"],
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics_json, file, indent=2, cls=NumpyJSONEncoder)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(str(metrics["classification_report_text"]))
