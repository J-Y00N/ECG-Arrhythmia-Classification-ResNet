"""Per-beat prediction table, the input to every post-hoc analysis.

Model selection and the headline metrics can be computed on the fly, but the
cluster bootstrap, the per-record decomposition and the effective-patient-count
analysis all need one row per beat with the recording it came from. Writing
that table at evaluation time means those analyses never require re-running
inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_classification.constants import CLASS_SYMBOLS, PREDICTION_COLUMNS


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    record_ids: np.ndarray,
    beat_indices: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Run inference and pair every prediction with its recording.

    The loader must be unshuffled and must not drop the final partial batch, so
    that its output order matches ``record_ids`` element for element. This is
    checked rather than assumed, because a silent misalignment here would
    corrupt every downstream patient-level result while leaving the aggregate
    metrics looking perfectly normal.
    """
    model.eval()

    probability_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    for signals, labels in dataloader:
        logits = model(signals.to(device))
        probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        label_batches.append(labels.numpy())

    probabilities = np.concatenate(probability_batches).astype(np.float32)
    true_labels = np.concatenate(label_batches).astype(np.int64)

    if len(probabilities) != len(record_ids):
        raise RuntimeError(
            f"prediction count {len(probabilities)} does not match record id count "
            f"{len(record_ids)}. The evaluation loader must use shuffle=False and "
            f"drop_last=False."
        )

    frame = pd.DataFrame(
        {
            "record_id": np.asarray(record_ids, dtype=str),
            "beat_index": np.asarray(beat_indices, dtype=np.int64),
            "y_true": true_labels,
            "y_pred": probabilities.argmax(axis=1).astype(np.int64),
        }
    )
    for position, symbol in enumerate(CLASS_SYMBOLS):
        frame[f"p_{symbol}"] = probabilities[:, position]

    return frame[list(PREDICTION_COLUMNS)]


def save_predictions(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Write the prediction table next to the run's other artefacts."""
    path = Path(output_dir) / "predictions.csv"
    frame.to_csv(path, index=False, float_format="%.6f")
    return path