"""Calibration, and what the model does when it meets pure artefact.

Accuracy says how often a prediction is right. Calibration says whether the
probability attached to it can be believed. The two come apart under
distribution shift, and they come apart in the direction that matters
clinically: a model can keep most of its accuracy while its confidence stops
tracking its correctness, so the failures it does have arrive unflagged.

Three things are computed.

**Reliability.** Predictions are binned by confidence and the accuracy within
each bin is compared with the confidence claimed. A calibrated model sits on the
diagonal; one that sits below it is overconfident. Expected calibration error is
the average gap, weighted by how many predictions fall in each bin.

**Per-recording calibration.** The same quantity computed for each test
recording. The aggregate can look acceptable while individual patients are badly
served, and the aggregate is not what a patient experiences.

**The observation set.** The unclassifiable beats -- fifteen across the whole
database, all of them baseline wander or electrode transients -- were held out
of training, validation and test. The model has never seen pure artefact. What
it does when it meets some is a calibration question, not a classification one:
fifteen beats cannot be scored, but a classifier that answers "normal beat" at
0.99 on a detached-electrode waveform fails in a way that matters more than one
that answers uncertainly.

Usage
-----
    python -m analysis.calibration outputs/wide187-inter-none-lossweight0-seed42
    python -m analysis.calibration RUN --compare RUN_INTRA
    python -m analysis.calibration RUN --bins 20 --figure docs/assets/result/reliability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_SYMBOLS = ("N", "S", "V", "F")
PROBABILITY_COLUMNS = tuple(f"p_{s}" for s in CLASS_SYMBOLS)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"no table at {path}")
    frame = pd.read_csv(path, dtype={"record_id": str})
    missing = [c for c in PROBABILITY_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} has no probability columns {missing}")
    return frame


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def reliability(
    confidence: np.ndarray, correct: np.ndarray, bins: int
) -> tuple[pd.DataFrame, float, float]:
    """Bin by confidence and compare it against accuracy.

    Equal-width bins over [1/K, 1] rather than [0, 1]: with K classes the
    softmax maximum cannot fall below 1/K, so bins under that are empty by
    construction and would dilute the average.

    Returns the per-bin table, the expected calibration error and the maximum
    calibration error. ECE is the mean gap weighted by bin population; MCE is
    the worst single bin, which is the one a clinician would meet.
    """
    floor = 1.0 / len(CLASS_SYMBOLS)
    edges = np.linspace(floor, 1.0, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)

    rows = []
    total = len(confidence)
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1], "n": 0,
                         "confidence": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        mean_confidence = float(confidence[mask].mean())
        mean_accuracy = float(correct[mask].mean())
        rows.append({
            "bin_low": edges[b], "bin_high": edges[b + 1], "n": count,
            "confidence": mean_confidence, "accuracy": mean_accuracy,
            "gap": mean_accuracy - mean_confidence,
        })

    table = pd.DataFrame(rows)
    populated = table[table.n > 0]
    ece = float((populated.n / total * populated.gap.abs()).sum())
    mce = float(populated.gap.abs().max()) if len(populated) else float("nan")
    return table, ece, mce


def report_calibration(frame: pd.DataFrame, bins: int, label: str) -> dict:
    probability = probabilities(frame)
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    correct = (predicted == frame["y_true"].to_numpy()).astype(float)

    table, ece, mce = reliability(confidence, correct, bins)

    print("\n" + "=" * 82)
    print(f"CALIBRATION  ({label})")
    print("=" * 82)
    print(f"  {len(frame):,} predictions, mean confidence {confidence.mean():.4f}, "
          f"accuracy {correct.mean():.4f}")
    print(f"  overconfidence (confidence - accuracy)   {confidence.mean() - correct.mean():+.4f}")
    print(f"  expected calibration error   ECE         {ece:.4f}")
    print(f"  maximum calibration error    MCE         {mce:.4f}")

    print(f"\n  {'confidence bin':>18}{'n':>9}{'claimed':>10}{'actual':>9}{'gap':>9}")
    print("  " + "-" * 55)
    for _, row in table.iterrows():
        if row.n == 0:
            continue
        print(f"  {row.bin_low:>8.2f} - {row.bin_high:<7.2f}{int(row.n):>9,}"
              f"{row.confidence:>10.3f}{row.accuracy:>9.3f}{row.gap:>+9.3f}")
    print("  A negative gap means the model claimed more than it delivered.")

    return {"table": table, "ece": ece, "mce": mce,
            "confidence": float(confidence.mean()), "accuracy": float(correct.mean())}


def report_per_record_calibration(frame: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Calibration one recording at a time.

    Reported because the aggregate is an average over patients and no patient
    experiences an average. A model can be well calibrated overall while being
    badly overconfident on the recordings it handles worst -- which is the
    combination that does damage, since those are the patients whose results
    most need a warning attached.
    """
    rows = []
    for record, group in frame.groupby("record_id"):
        probability = probabilities(group)
        confidence = probability.max(axis=1)
        correct = (probability.argmax(axis=1) == group["y_true"].to_numpy()).astype(float)
        _, ece, _ = reliability(confidence, correct, bins)
        rows.append({
            "record_id": record, "n": len(group),
            "confidence": float(confidence.mean()),
            "accuracy": float(correct.mean()),
            "overconfidence": float(confidence.mean() - correct.mean()),
            "ece": ece,
        })

    table = pd.DataFrame(rows).sort_values("overconfidence", ascending=False)

    print("\n" + "=" * 82)
    print("CALIBRATION BY RECORDING")
    print("=" * 82)
    print(f"  {'record':>8}{'n':>8}{'confidence':>12}{'accuracy':>10}{'over':>9}{'ECE':>8}")
    print("  " + "-" * 55)
    for _, row in table.iterrows():
        print(f"  {row.record_id:>8}{row.n:>8,}{row.confidence:>12.3f}"
              f"{row.accuracy:>10.3f}{row.overconfidence:>+9.3f}{row.ece:>8.3f}")

    worst = table.iloc[0]
    print(f"\n  worst served: {worst.record_id}, claiming {worst.confidence:.3f} "
          f"while delivering {worst.accuracy:.3f}")
    print("  The aggregate is an average over patients; no patient experiences it.")
    return table


# ---------------------------------------------------------------------------
# Observation set
# ---------------------------------------------------------------------------

def report_observations(run_dir: Path) -> None:
    """What the model does with beats the annotators could not classify.

    Not scored. Fifteen beats support no estimate, and there is no correct
    answer to compare against -- the label means "unclassifiable", not a class
    the network was asked to learn. The question is narrower: does a classifier
    that has only ever seen clean morphology hedge when it meets artefact, or
    does it answer with the same confidence it brings to a normal beat?
    """
    path = Path(run_dir) / "observations.csv"
    if not path.exists():
        print("\n(no observation set for this run)")
        return

    frame = load_table(path)
    probability = probabilities(frame)
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)

    print("\n" + "=" * 82)
    print("OBSERVATION SET: unclassifiable beats")
    print("=" * 82)
    print(f"  {len(frame)} beats, held out of training, validation and test.")
    print("  The annotators marked these as unclassifiable; they are baseline")
    print("  wander and electrode transients, not a class the model was taught.")

    print(f"\n  {'record':>8}{'predicted':>12}{'confidence':>12}   "
          + "".join(f"{s:>8}" for s in CLASS_SYMBOLS))
    print("  " + "-" * 66)
    for i, row in frame.iterrows():
        print(f"  {row.record_id:>8}{CLASS_SYMBOLS[predicted[i]]:>12}"
              f"{confidence[i]:>12.3f}   " + "".join(f"{p:>8.3f}" for p in probability[i]))

    counts = pd.Series([CLASS_SYMBOLS[p] for p in predicted]).value_counts()
    print("\n  predicted as: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  confidence: median {np.median(confidence):.3f}, "
          f"range {confidence.min():.3f} to {confidence.max():.3f}")

    high = (confidence > 0.9).sum()
    print(f"  {high} of {len(frame)} answered above 0.9")
    print("\n  A hedged answer here would be the desirable behaviour: the beat belongs")
    print("  to no class the model knows. High confidence on a detached-electrode")
    print("  waveform is the failure mode that matters clinically, because nothing")
    print("  downstream is told to doubt it.")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def save_reliability_figure(results: dict[str, dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    floor = 1.0 / len(CLASS_SYMBOLS)

    axes[0].plot([floor, 1], [floor, 1], "k--", lw=0.8, label="perfect calibration")
    for label, result in results.items():
        table = result["table"]
        populated = table[table.n > 0]
        axes[0].plot(populated.confidence, populated.accuracy, "o-", ms=4,
                     label=f"{label}  ECE {result['ece']:.3f}")
    axes[0].set_xlabel("claimed confidence")
    axes[0].set_ylabel("observed accuracy")
    axes[0].set_title("Reliability")
    axes[0].legend(fontsize=8)

    width = 0.8 / max(len(results), 1)
    for offset, (label, result) in enumerate(results.items()):
        table = result["table"]
        centres = (table.bin_low + table.bin_high) / 2
        axes[1].bar(centres + offset * width * 0.05, table.n.fillna(0),
                    width=(1 - floor) / len(table) * 0.9, alpha=0.55, label=label)
    axes[1].set_xlabel("claimed confidence")
    axes[1].set_ylabel("predictions")
    axes[1].set_yscale("log")
    axes[1].set_title("Where the predictions sit")
    axes[1].legend(fontsize=8)

    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    print(f"\nwrote {path}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, default=None,
                        help="a second run, usually the matching intra-patient one")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    def name(path: Path) -> str:
        config = Path(path) / "config.json"
        if config.exists():
            payload = json.loads(config.read_text(encoding="utf-8"))
            return payload.get("run_name", Path(path).name)
        return Path(path).name

    results = {}
    frame = load_table(Path(args.run) / "predictions.csv")
    results[name(args.run)] = report_calibration(frame, args.bins, name(args.run))
    report_per_record_calibration(frame, args.bins)
    report_observations(args.run)

    if args.compare is not None:
        other = load_table(Path(args.compare) / "predictions.csv")
        results[name(args.compare)] = report_calibration(other, args.bins, name(args.compare))

        print("\n" + "=" * 82)
        print("COMPARISON")
        print("=" * 82)
        print(f"  {'':<34}{'ECE':>9}{'overconfidence':>18}")
        for label in (name(args.run), name(args.compare)):
            r = results[label]
            print(f"  {label:<34}{r['ece']:>9.4f}{r['confidence'] - r['accuracy']:>+18.4f}")
        print("\n  Accuracy and calibration do not degrade together. A model can keep most")
        print("  of its accuracy under patient separation while its probabilities stop")
        print("  meaning what they claim, and the second failure is the quieter one.")

    if args.figure is not None:
        save_reliability_figure(results, args.figure)


if __name__ == "__main__":
    main()