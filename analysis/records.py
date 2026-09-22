"""Where the inter-patient penalty comes from, and what predicts it.

An aggregate score says how much was lost. It does not say whether the loss is
spread evenly across the test recordings or concentrated in a few, and the two
imply different things: a uniform penalty is a limit of the model, a
concentrated one is a property of particular patients.

Two views are produced.

**Per-record decomposition.** Accuracy and per-class F1 for each recording in
the test half, ordered worst first. This also supplies the visual case for the
cluster bootstrap: if recordings scatter widely, the recording is plainly the
unit that carries the variation.

**Effective patient count.** For each class, the inverse Simpson index over its
per-record distribution,

    N_eff(c) = 1 / sum_r p_(c,r)^2

which equals the number of contributing recordings when a class is spread
evenly and falls toward one when a single recording dominates. The prediction
tested here is that the per-class protocol penalty tracks N_eff: a class whose
training beats come from one patient has nothing to generalise from.

Both read the cached class-by-record table and the stored prediction tables.
Nothing is recomputed from a model.

Usage
-----
    python -m analysis.records outputs/wide187-inter-none-lossweight0-seed42
    python -m analysis.records RUN --intra RUN_INTRA
    python -m analysis.records RUN --data data --arm wide187
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_SYMBOLS = ("N", "S", "V", "F")
N_CLASSES = len(CLASS_SYMBOLS)
SELECTION = (0, 1, 2)

DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)
DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    flat = np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES**2)
    return flat.reshape(N_CLASSES, N_CLASSES).astype(np.float64)


def per_class_f1(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 2.0 * diagonal / denominator, np.nan)


def effective_records(counts: np.ndarray) -> float:
    """Inverse Simpson index over a class's per-record counts."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    shares = counts / total
    return float(1.0 / np.square(shares).sum())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"no prediction table at {path}")
    return pd.read_csv(path, dtype={"record_id": str})


def load_class_record_table(run_dir: Path, data_root: Path | None, arm: str | None) -> pd.DataFrame:
    """Find the class-by-record counts for the arm a run used.

    The recorded ``data_dir`` can be stale for runs made before the caches were
    gathered under one root, so the current layout is tried as well.
    """
    config_path = Path(run_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    arm = arm or config.get("representation", "narrow")

    candidates = [
        data_root,
        Path(config.get("data_dir", "data")),
        Path(run_dir).resolve().parents[1] / "data",
        Path("data"),
    ]
    for root in candidates:
        if root is None:
            continue
        path = Path(root) / arm / "class_record_counts.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"record_id": str})

    raise FileNotFoundError(
        f"no class_record_counts.csv for arm '{arm}' under "
        f"{[str(c) for c in candidates if c is not None]}"
    )


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def per_record_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record, group in frame.groupby("record_id"):
        matrix = confusion(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        f1 = per_class_f1(matrix)
        support = matrix.sum(axis=1)
        rows.append(
            {
                "record_id": record,
                "n": int(len(group)),
                "accuracy": float(np.diag(matrix).sum() / len(group)),
                **{f"f1_{s}": f1[i] for i, s in enumerate(CLASS_SYMBOLS)},
                **{f"n_{s}": int(support[i]) for i, s in enumerate(CLASS_SYMBOLS)},
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy").reset_index(drop=True)


def report_per_record(frame: pd.DataFrame) -> pd.DataFrame:
    table = per_record_table(frame)

    print("\n" + "=" * 84)
    print("PER-RECORD DECOMPOSITION")
    print("=" * 84)
    print(f"  accuracy over {len(table)} recordings: "
          f"median {table.accuracy.median():.3f}, "
          f"range {table.accuracy.min():.3f} to {table.accuracy.max():.3f}, "
          f"sd {table.accuracy.std(ddof=1):.3f}")

    print(f"\n  {'record':>7}{'n':>7}{'acc':>8}" + "".join(f"{'f1_' + s:>8}" for s in CLASS_SYMBOLS))
    print("  " + "-" * 62)
    for _, row in table.iterrows():
        cells = []
        for s in CLASS_SYMBOLS:
            value, support = row[f"f1_{s}"], row[f"n_{s}"]
            cells.append("      --" if support == 0 else f"{value:>8.3f}")
        print(f"  {row.record_id:>7}{row.n:>7}{row.accuracy:>8.3f}" + "".join(cells))
    print("  (-- means the recording contains no beats of that class)")

    worst = table.head(3)
    share = (1 - table.accuracy) * table.n
    concentration = share.sort_values(ascending=False).head(3).sum() / share.sum()
    print(f"\n  the three weakest recordings carry {concentration:.0%} of all errors")
    print("  weakest: " + ", ".join(
        f"{r.record_id} ({r.accuracy:.3f}, n={r.n:,})" for _, r in worst.iterrows()
    ))
    print("\n  A spread this wide is what the cluster bootstrap is for: the recording,")
    print("  not the beat, is the unit across which performance actually varies.")
    return table


def report_effective_records(counts: pd.DataFrame) -> pd.DataFrame:
    """Effective patient count per class, for the training and test halves."""
    print("\n" + "=" * 84)
    print("EFFECTIVE PATIENT COUNT")
    print("=" * 84)

    halves = {"DS1 (train)": DS1, "DS2 (test)": DS2, "all 44": tuple(counts.record_id)}
    rows = []
    for name, records in halves.items():
        subset = counts[counts.record_id.isin(records)]
        for symbol in CLASS_SYMBOLS:
            column = subset[symbol].to_numpy(dtype=float)
            rows.append(
                {
                    "half": name,
                    "class": symbol,
                    "beats": int(column.sum()),
                    "records_with_any": int((column > 0).sum()),
                    "n_eff": effective_records(column),
                    "top_record": subset.record_id.iloc[int(column.argmax())] if column.sum() else "-",
                    "top_share": float(column.max() / column.sum()) if column.sum() else 0.0,
                }
            )
    table = pd.DataFrame(rows)

    for name in halves:
        part = table[table.half == name]
        print(f"\n  {name}")
        print(f"    {'class':>6}{'beats':>9}{'records':>9}{'N_eff':>8}   dominant recording")
        for _, row in part.iterrows():
            print(f"    {row['class']:>6}{row.beats:>9,}{row.records_with_any:>9}"
                  f"{row.n_eff:>8.2f}   {row.top_record} ({row.top_share:.0%})")

    print("\n  N_eff is the number of recordings a class would need if its beats were")
    print("  spread evenly to carry the same concentration. A class at 1.2 has the")
    print("  beat count its frequency suggests and the patient count of a case report.")
    return table


def report_penalty_vs_neff(
    inter_frame: pd.DataFrame,
    intra_frame: pd.DataFrame,
    counts: pd.DataFrame,
) -> None:
    """Does the per-class protocol penalty track the effective patient count?"""
    inter_f1 = per_class_f1(confusion(inter_frame.y_true.to_numpy(), inter_frame.y_pred.to_numpy()))
    intra_f1 = per_class_f1(confusion(intra_frame.y_true.to_numpy(), intra_frame.y_pred.to_numpy()))

    train = counts[counts.record_id.isin(DS1)]

    print("\n" + "=" * 84)
    print("PENALTY AGAINST EFFECTIVE PATIENT COUNT")
    print("=" * 84)
    print(f"  {'class':>6}{'N_eff (DS1)':>13}{'intra F1':>10}{'inter F1':>10}{'penalty':>10}")
    print("  " + "-" * 51)

    neff, penalty = [], []
    for index, symbol in enumerate(CLASS_SYMBOLS):
        value = effective_records(train[symbol].to_numpy(dtype=float))
        drop = float(intra_f1[index] - inter_f1[index])
        neff.append(value)
        penalty.append(drop)
        print(f"  {symbol:>6}{value:>13.2f}{intra_f1[index]:>10.3f}"
              f"{inter_f1[index]:>10.3f}{drop:>10.3f}")

    neff, penalty = np.array(neff), np.array(penalty)
    finite = np.isfinite(neff) & np.isfinite(penalty)
    if finite.sum() >= 3:
        # Spearman on four points is a description, not a test. It is reported
        # because the direction is the claim and the magnitude is not.
        ranks_n = pd.Series(neff[finite]).rank().to_numpy()
        ranks_p = pd.Series(penalty[finite]).rank().to_numpy()
        rho = float(np.corrcoef(ranks_n, ranks_p)[0, 1])
        print(f"\n  Spearman rank correlation over {finite.sum()} classes: {rho:+.3f}")
        print("  Four points cannot support a p-value. The ordering is the observation:")
        print("  classes drawn from fewer patients lose more when patients are separated.")

    print("\n  Two mechanisms are at work and they are not the same. Ventricular beats")
    print("  survive on morphology, having a wide QRS complex that does not depend on")
    print("  knowing the patient. Supraventricular beats do not: their QRS is normal,")
    print("  they are defined by prematurity, and with timing absent from the")
    print("  representation only patient-specific morphology remains. Fusion beats fail")
    print("  on both counts at once.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="an inter-patient run")
    parser.add_argument("--intra", type=Path, default=None,
                        help="matching intra-patient run, for the penalty view")
    parser.add_argument("--data", type=Path, default=None, help="override the cache root")
    parser.add_argument("--arm", default=None, help="override the representation arm")
    parser.add_argument("--csv", type=Path, default=None, help="write the per-record table here")
    args = parser.parse_args()

    frame = load_predictions(args.run)
    counts = load_class_record_table(args.run, args.data, args.arm)

    print("=" * 84)
    print(Path(args.run).name)
    print("=" * 84)
    print(f"{len(frame):,} beats across {frame.record_id.nunique()} recordings")

    table = report_per_record(frame)
    report_effective_records(counts)

    if args.intra is not None:
        report_penalty_vs_neff(frame, load_predictions(args.intra), counts)

    if args.csv is not None:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()