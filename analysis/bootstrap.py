"""Uncertainty for patient-clustered predictions.

Beats are not independent observations. They sit inside recordings, and beats
from one recording share that patient's morphology, electrode placement, rate
and noise characteristics. A bootstrap that resamples beats therefore treats
roughly fifty thousand correlated observations as if they were fifty thousand
independent ones, and reports an interval far narrower than the data supports.

Two things are computed here, and they check each other.

The **cluster bootstrap** resamples recordings rather than beats: a recording
drawn twice contributes all of its beats twice. This answers the question
clinical practice actually asks -- *how would this model do on the next
patient?* -- rather than *how much would the score move if we re-drew beats from
these same twenty-two people?*, which is not a useful question once those
twenty-two have already been seen.

The **design effect** predicts, from the data alone, how much narrower the naive
interval should be. Writing the correctness of beat $j$ in recording $i$ as

    Y_ij = mu + a_i + e_ij,    a_i ~ (0, s2_a),  e_ij ~ (0, s2_e)

the intraclass correlation and design effect are

    rho  = s2_a / (s2_a + s2_e)
    Deff = 1 + (m_bar - 1) * rho
    n_eff = n / Deff

and the ratio of the two interval widths should come out near sqrt(Deff). When
it does, the argument for clustering stops being a claim and becomes a check.

Usage
-----
    python -m analysis.bootstrap outputs/wide187-inter-none-lossweight0-seed42
    python -m analysis.bootstrap RUN_A --compare RUN_B
    python -m analysis.bootstrap RUN --reps 5000 --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_SYMBOLS = ("N", "S", "V", "F")
N_CLASSES = len(CLASS_SYMBOLS)

#: Fusion cannot be scored under this protocol (see the report, section 3.6), so
#: the headline average runs over the other three. The four-class figure is
#: reported alongside rather than replaced.
SELECTION = (0, 1, 2)

#: Fraction of the smaller prediction table that must survive matching for a
#: paired comparison to mean anything. Representation arms lose a few dozen
#: beats at recording boundaries; protocols lose most of the table.
MIN_PAIRED_OVERLAP = 0.90


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Confusion matrix via bincount: fast enough to sit inside a bootstrap loop."""
    flat = np.bincount(
        y_true * N_CLASSES + y_pred, minlength=N_CLASSES * N_CLASSES
    )
    return flat.reshape(N_CLASSES, N_CLASSES).astype(np.float64)


def per_class_f1(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    denominator = predicted + actual
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(denominator > 0, 2.0 * diagonal / denominator, 0.0)
    return scores


def macro_f1(matrix: np.ndarray, classes: tuple[int, ...] = SELECTION) -> float:
    return float(per_class_f1(matrix)[list(classes)].mean())


# ---------------------------------------------------------------------------
# Intraclass correlation and design effect
# ---------------------------------------------------------------------------

def intraclass_correlation(correct: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects ICC on per-beat correctness, unbalanced groups.

    Uses the method of moments rather than REML: with a binary response and
    thousands of observations per group the two agree closely, and the moment
    estimator needs no optimiser and no distributional assumption beyond the
    two variance components.

    A negative variance estimate is possible when between-group variation is
    smaller than within-group noise; it is clipped to zero, which is the usual
    convention and means "no detectable clustering".
    """
    correct = np.asarray(correct, dtype=np.float64)
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)

    n_total = len(correct)
    n_groups = len(labels)
    if n_groups < 2:
        raise ValueError("intraclass correlation needs at least two groups")

    group_sums = np.bincount(inverse, weights=correct, minlength=n_groups)
    group_means = group_sums / counts
    grand_mean = correct.mean()

    ss_between = float((counts * (group_means - grand_mean) ** 2).sum())
    ss_within = float(((correct - group_means[inverse]) ** 2).sum())

    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)

    # Effective group size for unbalanced designs (Searle et al.). Equals the
    # common group size when the design is balanced, and is pulled below the
    # arithmetic mean when it is not.
    m_zero = (n_total - (counts.astype(np.float64) ** 2).sum() / n_total) / (n_groups - 1)

    var_between = max((ms_between - ms_within) / m_zero, 0.0)
    var_within = ms_within
    rho = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0

    mean_group_size = n_total / n_groups
    design_effect = 1.0 + (mean_group_size - 1.0) * rho

    return {
        "n_beats": n_total,
        "n_records": n_groups,
        "mean_group_size": mean_group_size,
        "m_zero": m_zero,
        "var_between": var_between,
        "var_within": var_within,
        "icc": rho,
        "design_effect": design_effect,
        "n_effective": n_total / design_effect if design_effect > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Bootstraps
# ---------------------------------------------------------------------------

def _group_indices(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.unique(groups)
    return labels, [np.flatnonzero(groups == label) for label in labels]


def cluster_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Resample recordings with replacement, carrying all of their beats.

    Also returns, for each recording, the fraction of replicates that contained
    it. That is not diagnostics for its own sake: a class supplied almost
    entirely by one recording is absent from roughly a third of replicates, and
    its interval is bimodal as a result. Knowing which recordings those are is
    what makes such an interval readable rather than puzzling.
    """
    labels, per_group = _group_indices(groups)
    n_groups = len(labels)

    macro3 = np.empty(reps)
    per_class = np.empty((reps, N_CLASSES))
    appearances = np.zeros(n_groups)

    for rep in range(reps):
        drawn = rng.integers(0, n_groups, size=n_groups)
        appearances[np.unique(drawn)] += 1
        index = np.concatenate([per_group[g] for g in drawn])
        matrix = confusion(y_true[index], y_pred[index])
        macro3[rep] = macro_f1(matrix)
        per_class[rep] = per_class_f1(matrix)

    coverage = {str(label): appearances[i] / reps for i, label in enumerate(labels)}
    return macro3, per_class, coverage


def naive_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample beats with replacement, ignoring which recording they came from.

    Computed only so that it can be shown to be wrong, and by how much.
    """
    n = len(y_true)
    macro3 = np.empty(reps)
    per_class = np.empty((reps, N_CLASSES))

    for rep in range(reps):
        index = rng.integers(0, n, size=n)
        matrix = confusion(y_true[index], y_pred[index])
        macro3[rep] = macro_f1(matrix)
        per_class[rep] = per_class_f1(matrix)

    return macro3, per_class


def align_predictions(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict two prediction tables to the beats both runs scored.

    Representation arms do not produce identical beat sets. A window reaching
    0.80 s either side cannot be centred as close to a recording's start or end
    as one reaching 0.25 s, so the wider arms drop a few dozen more beats at the
    boundaries. The difference is tiny -- 25 beats out of fifty thousand between
    ``narrow`` and ``wide187`` -- but a paired bootstrap has to score the same
    beats under both models, and pairing rows that are not the same beat would
    corrupt the comparison in a way no summary statistic would reveal.

    Beats are matched on ``(record_id, position)`` where both tables carry the
    R-peak position, and on ``(record_id, beat_index)`` otherwise. The number
    set aside is printed rather than absorbed silently: if it were ever large,
    the two runs would not be comparable and the caller should know.

    Raises if the matched labels disagree, which would mean the two tables
    describe different beats under the same key.
    """
    for column in ("record_id", "y_true", "y_pred"):
        if column not in frame_a.columns or column not in frame_b.columns:
            raise ValueError(f"both tables need a '{column}' column")

    if len(frame_a) == len(frame_b) and frame_a["record_id"].equals(frame_b["record_id"]):
        return frame_a.reset_index(drop=True), frame_b.reset_index(drop=True)

    key = ["record_id", "position"] if (
        "position" in frame_a.columns and "position" in frame_b.columns
    ) else ["record_id", "beat_index"]

    merged = frame_a.merge(frame_b, on=key, suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        raise ValueError(
            f"the two tables share no beats under {key}; they cannot be paired"
        )

    # A few dozen beats differ between representation arms, which is fine. A
    # large shortfall means something else: the two runs were evaluated on
    # different test sets, almost always because they use different protocols.
    # The intersection then holds beats that were unseen for one model and whose
    # *patient* was seen for the other, and scoring both on it answers no
    # question either of them was asked.
    overlap = len(merged) / min(len(frame_a), len(frame_b))
    if overlap < MIN_PAIRED_OVERLAP:
        raise ValueError(
            f"only {overlap:.0%} of the smaller table survives matching "
            f"({len(merged):,} of {min(len(frame_a), len(frame_b)):,} beats). The two "
            f"runs were not evaluated on the same test set, so a paired comparison is "
            f"not defined between them. Report their intervals side by side instead:\n"
            f"    python -m analysis.bootstrap RUN_A\n"
            f"    python -m analysis.bootstrap RUN_B\n"
            f"and read off whether the two intervals overlap."
        )
    if not (merged["y_true_a"].to_numpy() == merged["y_true_b"].to_numpy()).all():
        raise ValueError(
            f"matched rows disagree on the true label, so {key} is not identifying "
            f"the same beat in both tables.\n"
            f"If the key is beat_index, the two runs use different caches and their "
            f"indices are not comparable; the R-peak position is, and is recovered "
            f"automatically when the cache is reachable from the run's config.json."
        )

    print(
        f"  matched {len(merged):,} beats on {'+'.join(key)}; "
        f"set aside {len(frame_a) - len(merged)} from A and "
        f"{len(frame_b) - len(merged)} from B"
    )

    aligned_a = merged[key + ["y_true_a", "y_pred_a"]].rename(
        columns={"y_true_a": "y_true", "y_pred_a": "y_pred"}
    )
    aligned_b = merged[key + ["y_true_b", "y_pred_b"]].rename(
        columns={"y_true_b": "y_true", "y_pred_b": "y_pred"}
    )
    return aligned_a.reset_index(drop=True), aligned_b.reset_index(drop=True)


def paired_cluster_bootstrap(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Difference between two runs under the same resampled recordings.

    Pairing matters. Two runs evaluated on the same test half share the
    recordings that make the score move, so an unpaired comparison spends most
    of its variance on differences that cancel. Drawing one set of recordings
    and scoring both runs on it isolates the difference between the models.

    Beats are aligned first; see :func:`align_predictions` for why the two
    tables may not hold exactly the same rows.
    """
    aligned_a, aligned_b = align_predictions(frame_a, frame_b)

    groups = aligned_a["record_id"].to_numpy()
    labels, per_group = _group_indices(groups)
    n_groups = len(labels)

    true_a = aligned_a["y_true"].to_numpy()
    pred_a = aligned_a["y_pred"].to_numpy()
    pred_b = aligned_b["y_pred"].to_numpy()

    differences = np.empty(reps)
    for rep in range(reps):
        drawn = rng.integers(0, n_groups, size=n_groups)
        index = np.concatenate([per_group[g] for g in drawn])
        differences[rep] = (
            macro_f1(confusion(true_a[index], pred_a[index]))
            - macro_f1(confusion(true_a[index], pred_b[index]))
        )
    return differences


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.percentile(values, [100 * tail, 100 * (1 - tail)])
    return float(low), float(high)


#: Label held out of the model, and therefore out of the split indices that
#: ``beat_index`` refers to. Kept local so this module does not import the
#: training package.
OBSERVATION_LABEL = 4


def attach_position(frame: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    """Add the R-peak sample position to a prediction table, from its cache.

    ``beat_index`` is an index into the arm's own cache, and the arms do not
    hold the same beats: a window reaching 0.80 s either side cannot be centred
    as close to a recording's edge as one reaching 0.25 s, so the wider arms
    drop a few dozen more. Index 4,000 is therefore a different beat under
    ``narrow`` than under ``wide187``, and pairing on it would silently compare
    unrelated beats.

    The R-peak position is arm-independent -- it is a sample number in the
    original recording -- so it identifies a beat across arms. It is recovered
    here rather than stored at evaluation time, which would have meant repeating
    every run.

    Returns the frame unchanged if the cache cannot be located, leaving the
    caller to fall back and the guard in :func:`align_predictions` to catch any
    mismatch.
    """
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return frame

    config = json.loads(config_path.read_text(encoding="utf-8"))
    arm = config.get("representation", "narrow")

    # The recorded data_dir can be stale: runs made before the caches were
    # gathered under one root point at directories that no longer exist. Try
    # what the run says first, then the layout the project uses now.
    run_dir = Path(run_dir)
    candidates = [
        Path(config.get("data_dir", "data")),
        run_dir.resolve().parents[1] / "data",
        Path("data"),
    ]
    cache_path = None
    for root in candidates:
        candidate = root / arm / "beats.npz"
        if candidate.exists():
            cache_path = candidate
            break

    if cache_path is None:
        print(
            f"  warning: no cache for arm '{arm}' under any of "
            f"{[str(c) for c in candidates]}; beats cannot be matched on R-peak "
            f"position and the fallback key may not identify the same beats"
        )
        return frame

    with np.load(cache_path, allow_pickle=False) as archive:
        labels, positions = archive["y"], archive["position"]

    # The observation class is filtered out before the split is made, so the
    # indices in the table count only the modelled beats.
    modelled = positions[labels != OBSERVATION_LABEL]

    index = frame["beat_index"].to_numpy()
    if index.max() >= len(modelled):
        print(
            f"  warning: {cache_path} holds {len(modelled):,} modelled beats but the "
            f"table indexes up to {index.max():,}; the cache was rebuilt since this "
            f"run and positions cannot be recovered from it"
        )
        return frame

    out = frame.copy()
    out["position"] = modelled[index]
    return out


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"no prediction table at {path}")
    frame = pd.read_csv(path, dtype={"record_id": str})
    for column in ("record_id", "y_true", "y_pred"):
        if column not in frame.columns:
            raise ValueError(f"{path} has no '{column}' column")
    if "position" not in frame.columns and "beat_index" in frame.columns:
        frame = attach_position(frame, run_dir)
    return frame


def report_run(run_dir: Path, reps: int, rng: np.random.Generator) -> None:
    frame = load_predictions(run_dir)
    y_true = frame["y_true"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    groups = frame["record_id"].to_numpy()

    point = confusion(y_true, y_pred)
    point_f1 = per_class_f1(point)

    print("=" * 78)
    print(Path(run_dir).name)
    print("=" * 78)
    print(f"{len(y_true):,} beats across {len(np.unique(groups))} recordings")

    # --- design effect -----------------------------------------------------
    stats = intraclass_correlation((y_true == y_pred).astype(float), groups)
    print("\nCLUSTERING")
    print(f"  intraclass correlation   rho    = {stats['icc']:.4f}")
    print(f"  mean beats per recording m_bar  = {stats['mean_group_size']:.1f}")
    print(f"  design effect            Deff   = {stats['design_effect']:.2f}")
    print(f"  effective sample size    n_eff  = {stats['n_effective']:.0f}"
          f"   (of {stats['n_beats']:,} beats)")
    print(f"  predicted width ratio    sqrt   = {np.sqrt(stats['design_effect']):.2f}")

    # --- the two bootstraps ------------------------------------------------
    cluster_macro, cluster_class, coverage = cluster_bootstrap(
        y_true, y_pred, groups, reps, rng
    )
    naive_macro, naive_class = naive_bootstrap(y_true, y_pred, reps, rng)

    cl_low, cl_high = percentile_interval(cluster_macro)
    na_low, na_high = percentile_interval(naive_macro)
    naive_width = na_high - na_low
    # A degenerate width means every replicate scored identically, which happens
    # when a class is absent or a classifier is perfect. Reporting an infinite
    # ratio would be worse than reporting none.
    observed_ratio = (cl_high - cl_low) / naive_width if naive_width > 0 else float("nan")

    print("\nMACRO F1 (N/S/V)")
    print(f"  point estimate                  {macro_f1(point):.4f}")
    print(f"  beat bootstrap        95% CI    [{na_low:.4f}, {na_high:.4f}]"
          f"   width {na_high - na_low:.4f}")
    print(f"  cluster bootstrap     95% CI    [{cl_low:.4f}, {cl_high:.4f}]"
          f"   width {cl_high - cl_low:.4f}")
    ratio_text = "undefined" if np.isnan(observed_ratio) else f"{observed_ratio:.2f}"
    print(f"  observed width ratio            {ratio_text}"
          f"   (predicted {np.sqrt(stats['design_effect']):.2f})")
    print("  The prediction is first-order: Deff assumes equal group sizes and a")
    print("  linear statistic, and macro F1 is neither. Agreement to within a")
    print("  fraction is the check; exact equality is not expected.")

    print("\nPER CLASS, cluster bootstrap")
    print(f"  {'':>4}{'point':>9}{'95% CI':>22}{'width':>9}{'beat CI width':>16}")
    for index, symbol in enumerate(CLASS_SYMBOLS):
        low, high = percentile_interval(cluster_class[:, index])
        nlow, nhigh = percentile_interval(naive_class[:, index])
        print(f"  {symbol:>4}{point_f1[index]:>9.4f}"
              f"   [{low:>6.4f}, {high:>6.4f}]"
              f"{high - low:>9.4f}{nhigh - nlow:>16.4f}")

    # --- coverage, which explains any bimodal interval ---------------------
    sparse = sorted(coverage.items(), key=lambda kv: kv[1])[:3]
    print("\nRECORDING COVERAGE")
    print(f"  a recording appears in {np.mean(list(coverage.values())):.1%} of replicates on average")
    print("  absent most often: " + ", ".join(f"{r} ({1 - c:.0%} absent)" for r, c in sparse))
    print("  A class concentrated in one recording is undefined whenever that")
    print("  recording is not drawn, which is what makes its interval bimodal.")


def _protocol_of(run_dir: Path) -> str | None:
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8")).get("protocol")


def report_comparison(run_a: Path, run_b: Path, reps: int, rng: np.random.Generator) -> None:
    protocol_a, protocol_b = _protocol_of(run_a), _protocol_of(run_b)
    if protocol_a and protocol_b and protocol_a != protocol_b:
        raise ValueError(
            f"run A uses the {protocol_a} protocol and run B the {protocol_b} one, so "
            f"they were evaluated on different test sets and cannot be paired. The "
            f"protocol gap is read from their separate intervals, not from a paired "
            f"difference."
        )

    frame_a = load_predictions(run_a)
    frame_b = load_predictions(run_b)

    print("\n" + "=" * 78)
    print("PAIRED COMPARISON, cluster bootstrap")
    print("=" * 78)
    print(f"  A  {Path(run_a).name}")
    print(f"  B  {Path(run_b).name}")

    differences = paired_cluster_bootstrap(frame_a, frame_b, reps, rng)
    low, high = percentile_interval(differences)
    print(f"\n  macro F1 difference (A - B)   {differences.mean():+.4f}")
    print(f"  95% CI                        [{low:+.4f}, {high:+.4f}]")
    crosses = low <= 0.0 <= high
    print(f"  interval contains zero        {crosses}")
    print("\n  " + (
        "The two are not distinguishable once the recordings are treated as the\n"
        "  sampling unit."
        if crosses else
        "The difference survives treating the recordings as the sampling unit."
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    report_run(args.run, args.reps, rng)
    if args.compare is not None:
        report_comparison(args.run, args.compare, args.reps, rng)


if __name__ == "__main__":
    main()