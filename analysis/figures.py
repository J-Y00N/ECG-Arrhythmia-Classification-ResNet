"""Figures for the report, built from the stored artefacts.

Every panel reads a prediction table, a metrics file or the class-by-record
counts. Nothing retrains and nothing recomputes an inference pass, which is what
the prediction tables were written for.

Six figures, matching the sections that need them:

    result_protocol_gap        4.1   the headline, with cluster intervals
    result_bootstrap_widths    4.2   beat against record resampling
    result_per_record          4.4   where the penalty sits
    result_neff_penalty        4.3   penalty against effective patient count
    result_capacity            4.5   nearest neighbour, linear, convolutional
    result_representation      4.6   the arms, and the class they trade

Missing runs are skipped with a note rather than failing, so the script is
usable before the matrix is complete.

Usage
-----
    python -m analysis.figures
    python -m analysis.figures --only capacity representation
    python -m analysis.figures --reps 500      # faster, for a draft
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_SYMBOLS = ("N", "S", "V", "F")
SELECTION = (0, 1, 2)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Runs the figures draw on, by the name each is known by in the report.
RUNS = {
    "narrow inter": "inter-none-lossweight0-seed42",
    "narrow intra": "intra-none-lossweight0-seed42",
    "wide187 inter": "wide187-inter-none-lossweight0-seed42",
    "wide187 intra": "wide187-intra-none-lossweight0-seed42",
    "wide400 inter": "wide400-inter-none-lossweight0-seed42",
    "rr_ratio inter": "rr_ratio-inter-none-lossweight0-seed42",
    "wide187_rr inter": "wide187_rr-inter-none-lossweight0-seed42",
    "1-NN inter": "narrow-inter-1nn",
    "1-NN intra": "narrow-intra-1nn",
    "logistic inter": "narrow-inter-logistic",
    "logistic intra": "narrow-intra-logistic",
}

DS2 = ("100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 "
       "231 232 233 234").split()

DS1 = ("101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 "
       "215 220 223 230").split()


# ---------------------------------------------------------------------------

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def confusion(y_true, y_pred, k=len(CLASS_SYMBOLS)):
    return np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k).astype(float)


def per_class_f1(matrix):
    denominator = matrix.sum(0) + matrix.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 2 * np.diag(matrix) / denominator, 0.0)


def macro3(matrix):
    return float(per_class_f1(matrix)[list(SELECTION)].mean())


def load(name, outputs):
    path = Path(outputs) / RUNS[name] / "predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"record_id": str})


def cluster_ci(frame, reps, rng, statistic=macro3):
    groups = frame.record_id.to_numpy()
    labels = np.unique(groups)
    per_group = [np.flatnonzero(groups == label) for label in labels]
    y, p = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    values = np.empty(reps)
    for rep in range(reps):
        drawn = rng.integers(0, len(labels), len(labels))
        index = np.concatenate([per_group[g] for g in drawn])
        values[rep] = statistic(confusion(y[index], p[index]))
    return statistic(confusion(y, p)), np.percentile(values, [2.5, 97.5])


def beat_ci(frame, reps, rng, statistic=macro3):
    y, p = frame.y_true.to_numpy(), frame.y_pred.to_numpy()
    values = np.empty(reps)
    for rep in range(reps):
        index = rng.integers(0, len(y), len(y))
        values[rep] = statistic(confusion(y[index], p[index]))
    return np.percentile(values, [2.5, 97.5])


# ---------------------------------------------------------------------------

def figure_protocol_gap(data, out, reps, rng):
    """4.1 — the headline, with the intervals that make it a measurement."""
    plt = style()
    pairs = [("narrow", "narrow intra", "narrow inter"),
             ("wide187", "wide187 intra", "wide187 inter")]
    pairs = [(n, a, b) for n, a, b in pairs if data.get(a) is not None and data.get(b) is not None]
    if not pairs:
        return print("  skip protocol_gap: no matching intra/inter pair")

    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    for row, (name, intra, inter) in enumerate(pairs):
        for offset, (label, colour) in zip((0.16, -0.16), (("intra", "#4C72B0"), ("inter", "#C44E52"))):
            key = intra if label == "intra" else inter
            point, (low, high) = cluster_ci(data[key], reps, rng)
            axis.errorbar(point, row + offset, xerr=[[point - low], [high - point]],
                          fmt="o", ms=6, capsize=4, color=colour,
                          label=label if row == 0 else None)
            axis.text(high + 0.012, row + offset, f"{point:.3f}", va="center", fontsize=8)

    axis.set_yticks(range(len(pairs)))
    axis.set_yticklabels([p[0] for p in pairs])
    axis.set_xlabel("macro F1 over N, S, V   (95% cluster bootstrap)")
    axis.set_title("Separating patients costs about half the score")
    axis.set_xlim(0.45, 1.06)
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "result_protocol_gap.png")
    print("  wrote result_protocol_gap.png")


def figure_bootstrap_widths(data, out, reps, rng):
    """4.2 — the same statistic under two resampling units."""
    plt = style()
    # `or` on DataFrames raises: pandas refuses to guess what truthiness means
    # for a table. Pick explicitly.
    frame = data.get("wide187 inter")
    if frame is None:
        frame = data.get("narrow inter")
    if frame is None:
        return print("  skip bootstrap_widths: no inter run")

    point, (cl_lo, cl_hi) = cluster_ci(frame, reps, rng)
    na_lo, na_hi = beat_ci(frame, reps, rng)

    fig, axis = plt.subplots(figsize=(6.4, 2.8))
    for row, (label, low, high, colour) in enumerate(
        [("beat", na_lo, na_hi, "#8C8C8C"), ("recording", cl_lo, cl_hi, "#C44E52")]
    ):
        axis.errorbar(point, row, xerr=[[point - low], [high - point]],
                      fmt="o", ms=6, capsize=5, color=colour)
        axis.text(high + 0.008, row, f"width {high - low:.3f}", va="center", fontsize=8)

    axis.set_yticks([0, 1])
    axis.set_yticklabels(["beats\nresampled", "recordings\nresampled"])
    axis.set_ylim(-0.6, 1.6)
    axis.set_xlabel("macro F1 over N, S, V")
    axis.set_title("Same point estimate, two resampling units")
    fig.tight_layout()
    fig.savefig(out / "result_bootstrap_widths.png")
    print("  wrote result_bootstrap_widths.png")


def figure_per_record(data, out):
    """4.4 — the penalty is concentrated, not uniform."""
    plt = style()
    frame = data.get("narrow inter")
    if frame is None:
        return print("  skip per_record: no narrow inter run")

    rows = []
    for record, group in frame.groupby("record_id"):
        matrix = confusion(group.y_true.to_numpy(), group.y_pred.to_numpy())
        rows.append({"record": record, "n": len(group),
                     "accuracy": float(np.diag(matrix).sum() / len(group))})
    table = pd.DataFrame(rows).sort_values("accuracy")

    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    colours = ["#C44E52" if a < 0.95 else "#4C72B0" for a in table.accuracy]
    axis.bar(range(len(table)), table.accuracy, color=colours)
    axis.set_xticks(range(len(table)))
    axis.set_xticklabels(table.record, rotation=90, fontsize=7)
    axis.axhline(table.accuracy.median(), color="k", lw=0.7, ls="--")
    axis.text(len(table) - 0.5, table.accuracy.median() + 0.012,
              f"median {table.accuracy.median():.3f}", ha="right", fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("accuracy")
    axis.set_xlabel("test recording")
    axis.set_title("Three recordings carry two thirds of the errors")
    fig.tight_layout()
    fig.savefig(out / "result_per_record.png")
    print("  wrote result_per_record.png")


def figure_neff_penalty(data, out, counts):
    """4.3 — the penalty orders inversely with effective patient count."""
    plt = style()
    inter, intra = data.get("narrow inter"), data.get("narrow intra")
    if inter is None or intra is None or counts is None:
        return print("  skip neff_penalty: needs both protocols and the count table")

    f1_inter = per_class_f1(confusion(inter.y_true.to_numpy(), inter.y_pred.to_numpy()))
    f1_intra = per_class_f1(confusion(intra.y_true.to_numpy(), intra.y_pred.to_numpy()))
    train = counts[counts.record_id.isin(DS1)]

    neff, penalty = [], []
    for index, symbol in enumerate(CLASS_SYMBOLS):
        column = train[symbol].to_numpy(float)
        shares = column / column.sum()
        neff.append(1.0 / np.square(shares).sum())
        penalty.append(f1_intra[index] - f1_inter[index])

    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    # One colour per class, as in the other panels, so a reader who has seen
    # the representation figure can read this one without the labels.
    palette = {"N": "#4C72B0", "S": "#DD8452", "V": "#55A868", "F": "#C44E52"}
    axis.scatter(neff, penalty, s=70,
                 color=[palette[sym] for sym in CLASS_SYMBOLS], zorder=3)
    for x, y, symbol in zip(neff, penalty, CLASS_SYMBOLS):
        axis.annotate(f"  {symbol}  (N_eff {x:.2f})", (x, y), fontsize=8, va="center")
    axis.set_xscale("log")
    axis.set_xlabel("effective contributing records in DS1  (log scale)")
    axis.set_ylabel("intra F1 − inter F1")
    axis.set_title("Classes drawn from fewer patients lose more")
    axis.set_xlim(0.9, 40)
    axis.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    fig.savefig(out / "result_neff_penalty.png")
    print("  wrote result_neff_penalty.png")


def figure_capacity(data, out):
    """4.5 — the gap widens with how much a model can memorise."""
    plt = style()
    models = [("1-NN", "1-NN intra", "1-NN inter", "memorisation only"),
              ("CNN", "narrow intra", "narrow inter", "convolutional"),
              ("logistic", "logistic intra", "logistic inter", "linear")]
    models = [m for m in models if data.get(m[1]) is not None and data.get(m[2]) is not None]
    if not models:
        return print("  skip capacity: baselines not run")

    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    width = 0.36
    for row, (name, intra, inter, _) in enumerate(models):
        a = macro3(confusion(data[intra].y_true.to_numpy(), data[intra].y_pred.to_numpy()))
        b = macro3(confusion(data[inter].y_true.to_numpy(), data[inter].y_pred.to_numpy()))
        axis.barh(row + width / 2, a, height=width, color="#4C72B0",
                  label="intra" if row == 0 else None)
        axis.barh(row - width / 2, b, height=width, color="#C44E52",
                  label="inter" if row == 0 else None)
        axis.text(a + 0.008, row + width / 2, f"{a:.3f}", va="center", fontsize=8)
        axis.text(b + 0.008, row - width / 2, f"{b:.3f}", va="center", fontsize=8)
        axis.text(1.10, row, f"gap {a - b:.3f}", va="center", fontsize=8, color="#555555")

    axis.set_yticks(range(len(models)))
    axis.set_yticklabels([f"{m[0]}\n{m[3]}" for m in models])
    axis.set_xlim(0, 1.28)
    axis.set_xlabel("macro F1 over N, S, V")
    axis.set_title("Nearest neighbour reaches 0.92 when the patients are shared")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "result_capacity.png")
    print("  wrote result_capacity.png")


def figure_representation(data, out):
    """4.6 — the arms trade one ectopic class against the other."""
    plt = style()
    arms = ["narrow inter", "wide187 inter", "wide400 inter",
            "rr_ratio inter", "wide187_rr inter"]
    arms = [a for a in arms if data.get(a) is not None]
    if len(arms) < 2:
        return print("  skip representation: fewer than two arms run")

    scores = {}
    for arm in arms:
        f1 = per_class_f1(confusion(data[arm].y_true.to_numpy(), data[arm].y_pred.to_numpy()))
        scores[arm.replace(" inter", "")] = f1

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = list(scores)
    x = np.arange(len(names))

    for index, (symbol, colour) in enumerate(
        zip(CLASS_SYMBOLS, ["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ):
        axes[0].plot(x, [scores[n][index] for n in names], "o-", ms=5,
                     color=colour, label=symbol)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(-0.04, 1.04)
    axes[0].set_title("Per class, by representation")
    axes[0].legend(ncol=4, loc="center left")

    total = [scores[n][1] + scores[n][2] for n in names]
    axes[1].bar(x - 0.18, [scores[n][1] for n in names], width=0.36,
                color="#DD8452", label="S")
    axes[1].bar(x + 0.18, [scores[n][2] for n in names], width=0.36,
                color="#55A868", label="V")
    axes[1].plot(x, total, "k.--", ms=7, lw=1, label="S + V")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.2)
    axes[1].set_title("The two ectopic classes, and their sum")
    axes[1].legend(ncol=3)

    fig.tight_layout()
    fig.savefig(out / "result_representation.png")
    print("  wrote result_representation.png")



# ---------------------------------------------------------------------------
# Method figures
# ---------------------------------------------------------------------------

def figure_architecture(out):
    """3.2 - the network, and its size.

    The only figure here built from the model object rather than from a stored
    artefact, because it reports the parameter count. It is drawn for whichever
    arm the process was launched under, so the input shape and the classifier
    width follow ECG_REPRESENTATION and ECG_KEEP_Q.
    """
    from ecg_classification.constants import (
        BEAT_LENGTH, INPUT_CHANNELS, MODEL_INTERVAL_FEATURES, NUM_CLASSES,
        REPRESENTATION,
    )
    from ecg_classification.model import ResidualCNN

    plt = style()
    model = ResidualCNN(num_classes=NUM_CLASSES,
                        n_interval_features=MODEL_INTERVAL_FEATURES)
    parameters = sum(p.numel() for p in model.parameters())
    classifier_input = 32 + MODEL_INTERVAL_FEATURES

    blocks = [
        ("Input", f"{INPUT_CHANNELS} x {BEAT_LENGTH}"),
        ("Stem", "Conv1d(k=5)\nBatchNorm1d\nReLU"),
        ("Residual stack", "5 x ResidualBlock1D\nconv, conv, skip\nMaxPool(k=5, s=2)\nDropout 0.10"),
        ("Pooling", "AdaptiveAvgPool1d(1)"),
        ("Classifier", f"Linear({classifier_input}->64)\nReLU\nDropout 0.20\nLinear(64->{NUM_CLASSES})"),
        ("Output", f"{NUM_CLASSES} logits"),
    ]

    fig, axis = plt.subplots(figsize=(13, 2.9))
    axis.axis("off")
    width, gap, y = 0.138, 0.024, 0.34
    for index, (title, subtitle) in enumerate(blocks):
        x = 0.03 + index * (width + gap)
        axis.add_patch(plt.Rectangle((x, y), width, 0.46, facecolor="#e8f1fb",
                                     edgecolor="#2a5c8a", linewidth=1.4))
        axis.text(x + width / 2, y + 0.37, title, ha="center", va="center",
                  fontsize=10, fontweight="bold")
        axis.text(x + width / 2, y + 0.17, subtitle, ha="center", va="center", fontsize=7.8)
        if index < len(blocks) - 1:
            axis.annotate("", xy=(x + width + gap * 0.75, y + 0.23),
                          xytext=(x + width, y + 0.23),
                          arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#444"})

    axis.text(0.03, 0.18,
              f"{REPRESENTATION} arm.  {parameters:,} trainable parameters.  "
              "The convolutional trunk is identical in every arm; only the "
              "classifier's first layer changes width.",
              fontsize=8.5, va="top", ha="left")
    axis.set_xlim(0, 1)
    axis.set_ylim(0.1, 0.88)
    fig.savefig(out / "method_model_architecture.png", bbox_inches="tight")
    print("  wrote method_model_architecture.png")



def figure_protocols(out):
    """3.1 - how the two protocols divide the 44 recordings.

    The design in one panel. Under inter a recording belongs entirely to one
    side; under intra every recording is cut three ways. That difference is the
    whole experiment, and it is easier to see than to describe.
    """
    plt = style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.4), sharex=True)
    order = sorted(DS1 + DS2)
    x = np.arange(len(order))
    colours = {"train": "#4C72B0", "valid": "#DD8452", "test": "#55A868"}

    # seed 42 at a validation size of 0.20
    validation = {"122", "208", "209", "230"}
    role = ["valid" if r in validation else ("train" if r in DS1 else "test")
            for r in order]
    axes[0].bar(x, 1, color=[colours[r] for r in role], width=0.86)
    axes[0].set_ylabel("inter", rotation=0, ha="right", va="center")

    # every recording split at the beat level, in the same proportions
    for offset, (share, key) in enumerate(zip((0.72, 0.08, 0.20),
                                              ("train", "valid", "test"))):
        axes[1].bar(x, share, bottom=sum((0.72, 0.08, 0.20)[:offset]),
                    color=colours[key], width=0.86)
    axes[1].set_ylabel("intra", rotation=0, ha="right", va="center")

    for axis in axes:
        axis.set_yticks([])
        axis.set_xlim(-0.6, len(order) - 0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=90, fontsize=6)
    axes[1].set_xlabel("recording")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    axes[0].legend(handles, list(colours), ncol=3, loc="upper right",
                   bbox_to_anchor=(1.0, 1.75))
    fig.tight_layout()
    fig.savefig(out / "method_protocols.png")
    print("  wrote method_protocols.png")


def figure_windows(out):
    """3.1 - what each representation arm cuts out, on a shared time axis.

    Drawn against a schematic rhythm at 75 beats per minute so that the reader
    can see which arms admit a neighbouring R peak, and read off the resolution
    each pays for it.
    """
    plt = style()
    fig, axis = plt.subplots(figsize=(10, 3.0))
    arms = [("narrow", -0.248, 0.500, 187, 250),
            ("wide187", -0.800, 0.800, 187, 117),
            ("wide400", -0.800, 0.800, 400, 250)]

    for beat in (-1, 0, 1):
        centre = beat * 0.8
        t = np.linspace(centre - 0.08, centre + 0.08, 160)
        axis.plot(t, 0.30 * np.exp(-((t - centre) / 0.011) ** 2) - 0.52,
                  color="0.35", lw=1.0)
    axis.axhline(-0.52, color="0.8", lw=0.6, zorder=0)
    axis.text(-1.16, -0.52, "signal", fontsize=8, va="center", ha="right", color="0.35")

    for row, (name, start, stop, samples, rate) in enumerate(arms):
        y = row * 0.30
        axis.barh(y, stop - start, left=start, height=0.19, color="#4C72B0", alpha=0.25)
        axis.plot([0, 0], [y - 0.095, y + 0.095], color="#C44E52", lw=1.5)
        axis.text(-1.16, y, name, fontsize=9, va="center", ha="right")
        axis.text(stop + 0.05, y, f"{samples} samples, {rate} Hz", fontsize=8, va="center")

    axis.set_xlim(-1.2, 1.55)
    axis.set_ylim(-0.72, 0.80)
    axis.set_yticks([])
    axis.set_xlabel("seconds from the R peak")
    fig.tight_layout()
    fig.savefig(out / "method_windows.png")
    print("  wrote method_windows.png")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "assets" / "result")
    parser.add_argument("--method-out", type=Path,
                        default=PROJECT_ROOT / "docs" / "assets" / "method")
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="+", default=None,
                        choices=["gap", "widths", "records", "neff", "capacity",
                                 "representation", "protocols", "windows",
                                 "architecture"])
    args = parser.parse_args()

    data = {name: load(name, args.outputs) for name in RUNS}
    available = [n for n, f in data.items() if f is not None]
    print(f"{len(available)} of {len(RUNS)} runs found")
    for name in RUNS:
        if data[name] is None:
            print(f"  missing: {name}  ({RUNS[name]})")

    counts_path = args.data / "narrow" / "class_record_counts.csv"
    counts = pd.read_csv(counts_path, dtype={"record_id": str}) if counts_path.exists() else None

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    wanted = set(args.only) if args.only else None

    print()
    if wanted is None or "gap" in wanted:
        figure_protocol_gap(data, args.out, args.reps, rng)
    if wanted is None or "widths" in wanted:
        figure_bootstrap_widths(data, args.out, args.reps, rng)
    if wanted is None or "records" in wanted:
        figure_per_record(data, args.out)
    if wanted is None or "neff" in wanted:
        figure_neff_penalty(data, args.out, counts)
    if wanted is None or "capacity" in wanted:
        figure_capacity(data, args.out)
    if wanted is None or "representation" in wanted:
        figure_representation(data, args.out)

    args.method_out.mkdir(parents=True, exist_ok=True)
    if wanted is None or "protocols" in wanted:
        figure_protocols(args.method_out)
    if wanted is None or "windows" in wanted:
        figure_windows(args.method_out)
    if wanted is None or "architecture" in wanted:
        figure_architecture(args.method_out)


if __name__ == "__main__":
    main()