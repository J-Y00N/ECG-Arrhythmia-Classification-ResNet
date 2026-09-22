"""Exploratory figures for the representation arms.

The narrow arm was inspected visually before any training. The wide arms were
not, and their results turn on things that are easier to see than to infer: how
much of a neighbouring beat the window actually admits, and what dropping the
effective rate to 117 Hz does to a QRS complex.

Three views, each answering a question the numbers raise but do not settle.

**Class means, side by side.** One panel per arm, one line per class. Whether
the neighbouring beats are visible at all, and whether the complex survives the
resampling.

**The same beat under each arm.** Matched on R-peak position, so the three
traces are the same cardiac event seen through three representations. This is
the view that shows what 117 Hz costs, because the loss is only legible against
the 250 Hz version of the same beat.

**Neighbour visibility.** The fraction of beats whose previous R peak lands
inside the window, by class. The wide arms exist to admit that peak; this says
for how many beats they succeed.

Usage
-----
    python -m analysis.eda_arms
    python -m analysis.eda_arms --arms narrow wide187 wide400
    python -m analysis.eda_arms --record 232 --out docs/assets/eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASS_SYMBOLS = ("N", "S", "V", "F", "Q")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = ("narrow", "wide187", "wide400")


def load_arm(data_root: Path, arm: str) -> dict:
    """Load one arm's cache along with the window geometry it was built under.

    The manifest is the only reliable source for the geometry: the constants
    module reports whichever arm the current process was launched with, and this
    script deliberately reads several at once.
    """
    directory = Path(data_root) / arm
    cache_path = directory / "beats.npz"
    manifest_path = directory / "cache_manifest.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"no cache for arm '{arm}' at {cache_path}")

    with np.load(cache_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in ("x", "y", "record_id", "position")}

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    length = payload["x"].shape[1]
    window = manifest.get("window_length", length)
    payload.update(
        {
            "arm": arm,
            "window_length": window,
            "input_length": length,
            "pre_samples": manifest.get("pre_samples", 62),
            "target_fs": manifest.get("target_fs", 250),
            "effective_fs": manifest.get("effective_fs", 250.0),
            # Where the R peak sits in the model input, after any resampling.
            "r_peak_index": round(
                manifest.get("pre_samples", 62) * length / max(window, 1)
            ),
        }
    )
    return payload


def describe(arm: dict) -> str:
    seconds = arm["window_length"] / arm["target_fs"]
    return (
        f"{arm['arm']}  {seconds:.2f}s window, {arm['input_length']} samples, "
        f"{arm['effective_fs']:.0f} Hz, R at {arm['r_peak_index']}"
    )


# ---------------------------------------------------------------------------

def figure_class_means(arms: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4), squeeze=False)
    for column, arm in enumerate(arms):
        axis = axes[0][column]
        x, y = arm["x"], arm["y"]
        # A shared time axis in seconds, so the panels are comparable despite
        # holding different numbers of samples.
        seconds = (
            np.arange(arm["input_length"]) - arm["r_peak_index"]
        ) / (arm["input_length"] / (arm["window_length"] / arm["target_fs"]))

        for index, symbol in enumerate(CLASS_SYMBOLS[:4]):
            mask = y == index
            if mask.sum() < 5:
                continue
            axis.plot(seconds, x[mask].mean(axis=0), lw=1.2, label=f"{symbol} (n={mask.sum():,})")
        axis.axvline(0.0, color="k", lw=0.6, ls="--")
        axis.set_title(describe(arm), fontsize=9)
        axis.set_xlabel("seconds from R peak")
        if column == 0:
            axis.set_ylabel("normalised amplitude")
        axis.legend(fontsize=7)

    figure.tight_layout()
    path = out / "eda_arm_class_means.png"
    figure.savefig(path, dpi=140)
    print(f"  wrote {path}")


def figure_same_beat(arms: list[dict], out: Path, record: str, n_beats: int) -> None:
    """The same cardiac events under each arm, matched on R-peak position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Positions present in every arm; the wider windows drop a few at the edges.
    common = None
    for arm in arms:
        mask = arm["record_id"] == record
        positions = set(arm["position"][mask].tolist())
        common = positions if common is None else (common & positions)
    if not common:
        counts = ", ".join(
            f"{a['arm']} {(a['record_id'] == record).sum()}" for a in arms
        )
        raise ValueError(
            f"record {record} shares no R-peak positions across the arms ({counts} "
            f"beats each). Positions come from the annotation file and should be "
            f"identical wherever the arms both kept a beat, so an empty "
            f"intersection means the caches were built from different sources."
        )

    chosen = sorted(common)[len(common) // 3:][:n_beats]

    figure, axes = plt.subplots(
        len(chosen), len(arms), figsize=(4.6 * len(arms), 2.2 * len(chosen)), squeeze=False
    )
    for row, position in enumerate(chosen):
        for column, arm in enumerate(arms):
            axis = axes[row][column]
            mask = (arm["record_id"] == record) & (arm["position"] == position)
            index = int(np.flatnonzero(mask)[0])
            trace = arm["x"][index]
            seconds = (
                np.arange(len(trace)) - arm["r_peak_index"]
            ) / (len(trace) / (arm["window_length"] / arm["target_fs"]))

            axis.plot(seconds, trace, lw=0.9)
            axis.plot(seconds, trace, ".", ms=1.6, alpha=0.45)
            axis.axvline(0.0, color="r", lw=0.6)
            if row == 0:
                axis.set_title(describe(arm), fontsize=8)
            if column == 0:
                label = CLASS_SYMBOLS[int(arm["y"][index])]
                axis.set_ylabel(f"{record}  {label}", fontsize=8)
            axis.tick_params(labelsize=6)

    figure.suptitle(
        "The same beats under each arm. Dots are samples: their spacing is what "
        "the resampling changed.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / f"eda_arm_same_beat_{record}.png"
    figure.savefig(path, dpi=140)
    print(f"  wrote {path}")


def report_neighbour_visibility(arms: list[dict]) -> None:
    """How often the previous R peak falls inside each arm's window."""
    print("\n" + "=" * 78)
    print("NEIGHBOUR VISIBILITY")
    print("=" * 78)
    print("  fraction of beats whose previous R peak lands inside the window\n")
    print(f"  {'arm':>10}{'reach':>9}" + "".join(f"{s:>9}" for s in CLASS_SYMBOLS[:4]))
    print("  " + "-" * 55)

    for arm in arms:
        reach = arm["pre_samples"] / arm["target_fs"]
        shares = []
        for index in range(4):
            visible, total = 0, 0
            for record in np.unique(arm["record_id"]):
                mask = arm["record_id"] == record
                positions = np.sort(arm["position"][mask])
                labels = arm["y"][mask][np.argsort(arm["position"][mask])]
                if len(positions) < 2:
                    continue
                gaps = np.diff(positions) / arm["target_fs"]
                target = labels[1:] == index
                total += int(target.sum())
                visible += int((gaps[target] <= reach).sum())
            shares.append(visible / total if total else np.nan)
        print(f"  {arm['arm']:>10}{reach:>8.3f}s" + "".join(f"{v:>9.1%}" for v in shares))

    print("\n  A wide arm earns its resolution cost only if this gap between the")
    print("  supraventricular and normal columns is large: the previous peak has to")
    print("  be visible more often for ectopic beats than for normal ones.")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--record", default="232",
                        help="record for the matched-beat figure; 232 is the "
                             "recording the models handle worst")
    parser.add_argument("--n-beats", type=int, default=4)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "assets" / "eda")
    args = parser.parse_args()

    arms = []
    for name in args.arms:
        try:
            arms.append(load_arm(args.data, name))
        except FileNotFoundError as error:
            print(f"  skipping: {error}")
    if not arms:
        raise SystemExit("no arms could be loaded")

    print("=" * 78)
    print("REPRESENTATION ARMS")
    print("=" * 78)
    for arm in arms:
        print(f"  {describe(arm)}   {len(arm['x']):,} beats")

    report_neighbour_visibility(arms)

    args.out.mkdir(parents=True, exist_ok=True)
    print()
    figure_class_means(arms, args.out)
    figure_same_beat(arms, args.out, args.record, args.n_beats)


if __name__ == "__main__":
    main()