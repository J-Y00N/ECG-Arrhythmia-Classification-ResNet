"""Beats aligned on the *previous* R peak.

Every figure so far centres the current R peak, which is what the pipeline feeds
the network. That alignment makes supraventricular and normal beats look almost
identical, and the report says so in numbers -- normalised prematurity of 0.763
against 1.028 -- without ever showing it.

Re-anchoring on the preceding R peak shows it directly. The current beat's QRS
then lands at a distance equal to its preceding interval, so a premature beat
sits visibly earlier than a normal one and the class means separate along the
time axis rather than in amplitude.

Two panels, and the pair repeats a point the report makes elsewhere.

**Absolute time.** Seconds from the previous R peak. Separation appears but is
smeared, because a beat that is early for a slow patient is on time for a fast
one and both are averaged together.

**Normalised time.** The same beats with the axis divided by each recording's
local mean interval, so 1.0 is that patient's own resting interval. The
separation sharpens. This is the amplitude/IQR and beat-count/N_eff argument a
third time: absolute scales carry the patient, ratios carry the beat.

This is a diagnostic, not a representation. The network pools globally over
time, so the absolute position of a feature is not something it reads; what it
can read is the separation between two features, which is what the wide-window
arms supply and what section 4.6 measures.

Built from the ``wide400`` cache, whose 0.8 s reach already contains the
previous R peak for 98.6% of supraventricular beats.

Usage
-----
    python -m analysis.eda_prematurity
    python -m analysis.eda_prematurity --data data --out docs/assets/eda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASS_SYMBOLS = ("N", "S", "V", "F")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Beats either side used for the local mean interval, matching the pipeline.
LOCAL_WINDOW = 11

#: Where the previous R peak sits in the re-anchored view, in samples.
ANCHOR = 30

#: Length of the re-anchored view, in samples at the cache's rate.
VIEW_LENGTH = 340

#: Length of the normalised view, and how many local intervals it spans.
NORMALISED_LENGTH = 300
NORMALISED_SPAN = 1.6


def load(data_root: Path, arm: str) -> dict:
    directory = Path(data_root) / arm
    with np.load(directory / "beats.npz", allow_pickle=False) as archive:
        payload = {k: archive[k] for k in ("x", "y", "record_id", "position")}
    manifest = json.loads((directory / "cache_manifest.json").read_text(encoding="utf-8"))
    payload["fs"] = manifest["target_fs"]
    payload["r_index"] = manifest["pre_samples"]
    payload["arm"] = arm
    return payload


def interval_table(position: np.ndarray, record_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preceding interval and local mean interval, in samples, per beat.

    Computed per recording and in R-peak order; an interval across a recording
    boundary would be meaningless. The first beat of each recording has no
    preceding interval and is marked missing.
    """
    pre = np.full(len(position), np.nan)
    local = np.full(len(position), np.nan)

    for record in np.unique(record_id):
        mask = np.flatnonzero(record_id == record)
        order = mask[np.argsort(position[mask])]
        peaks = position[order].astype(np.float64)
        if len(peaks) < 3:
            continue

        gaps = np.diff(peaks)
        beat_pre = np.concatenate([[np.nan], gaps])

        half = LOCAL_WINDOW // 2
        padded = np.pad(gaps, half, mode="edge")
        kernel = np.ones(LOCAL_WINDOW) / LOCAL_WINDOW
        mean_gap = np.convolve(padded, kernel, mode="valid")[: len(gaps)]

        pre[order] = beat_pre
        local[order] = np.concatenate([[mean_gap[0]], mean_gap])

    return pre, local


def reanchor_absolute(cache: dict, pre: np.ndarray) -> np.ndarray:
    """Shift each beat so the previous R peak sits at ``ANCHOR``.

    Beats whose previous peak falls outside the cached window are dropped;
    tails that run past the cached window are left missing rather than padded,
    so the class mean is taken only over beats that actually reach that far.
    """
    x, r_index = cache["x"], cache["r_index"]
    out = np.full((len(x), VIEW_LENGTH), np.nan, dtype=np.float64)

    usable = np.isfinite(pre) & (pre <= r_index) & (pre >= 1)
    for index in np.flatnonzero(usable):
        start = int(r_index - pre[index]) - ANCHOR
        if start < 0:
            continue
        segment = x[index, start:]
        take = min(len(segment), VIEW_LENGTH)
        out[index, :take] = segment[:take]

    return out


def reanchor_normalised(cache: dict, pre: np.ndarray, local: np.ndarray) -> np.ndarray:
    """The same view with the time axis divided by the local mean interval.

    Each beat's segment is stretched or compressed so that one local mean
    interval occupies a fixed number of samples. A beat that is early for its
    own patient then lands in the same place regardless of that patient's rate,
    which is the whole point of the panel.
    """
    x, r_index = cache["x"], cache["r_index"]
    out = np.full((len(x), NORMALISED_LENGTH), np.nan, dtype=np.float64)
    target = np.linspace(0.0, NORMALISED_SPAN, NORMALISED_LENGTH)

    usable = np.isfinite(pre) & np.isfinite(local) & (pre <= r_index) & (local >= 1)
    for index in np.flatnonzero(usable):
        start = int(r_index - pre[index])
        if start < 0:
            continue
        segment = x[index, start:]
        if len(segment) < 10:
            continue
        # Position of each cached sample, measured in local mean intervals.
        source = np.arange(len(segment)) / local[index]
        inside = target <= source[-1]
        out[index, inside] = np.interp(target[inside], source, segment)

    return out


def report(cache: dict, pre: np.ndarray, local: np.ndarray) -> None:
    print("=" * 76)
    print(f"PREMATURITY, from the {cache['arm']} cache")
    print("=" * 76)
    print(f"  {'class':>6}{'beats':>9}{'pre-RR (s)':>13}{'/ local mean':>15}{'usable':>9}")
    print("  " + "-" * 52)

    for index, symbol in enumerate(CLASS_SYMBOLS):
        mask = (cache["y"] == index) & np.isfinite(pre) & np.isfinite(local)
        if not mask.any():
            continue
        usable = mask & (pre <= cache["r_index"])
        print(f"  {symbol:>6}{int(mask.sum()):>9,}"
              f"{np.median(pre[mask]) / cache['fs']:>13.3f}"
              f"{np.median(pre[mask] / local[mask]):>15.3f}"
              f"{usable.sum() / mask.sum():>9.1%}")

    print("\n  'usable' is the share whose previous R peak lies inside the cached")
    print("  window and can therefore be re-anchored. The ratio column is what the")
    print("  second panel plots against.")


def figure(cache: dict, absolute: np.ndarray, normalised: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    seconds = (np.arange(VIEW_LENGTH) - ANCHOR) / cache["fs"]
    ratios = np.linspace(0.0, NORMALISED_SPAN, NORMALISED_LENGTH)

    for index, symbol in enumerate(CLASS_SYMBOLS):
        mask = cache["y"] == index
        if mask.sum() < 20:
            continue

        for axis, grid, panel in (
            (axes[0], seconds, absolute),
            (axes[1], ratios, normalised),
        ):
            rows = panel[mask]
            enough = np.isfinite(rows).sum(axis=0) >= 20
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(rows, axis=0)
            mean[~enough] = np.nan
            axis.plot(grid, mean, lw=1.3, label=f"{symbol} (n={int(mask.sum()):,})")

    axes[0].axvline(0.0, color="k", lw=0.7, ls="--")
    axes[0].set_xlabel("seconds from the previous R peak")
    axes[0].set_ylabel("normalised amplitude")
    axes[0].set_title("absolute time — the patient's rate is mixed in", fontsize=10)

    axes[1].axvline(0.0, color="k", lw=0.7, ls="--")
    axes[1].axvline(1.0, color="k", lw=0.7, ls=":")
    axes[1].set_xlabel("intervals from the previous R peak (1.0 = that patient's local mean)")
    axes[1].set_title("normalised time — the beat alone", fontsize=10)

    for axis in axes:
        axis.legend(fontsize=8)

    fig.suptitle(
        "Beats aligned on the previous R peak. R-centred windows hide this "
        "separation entirely.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out.mkdir(parents=True, exist_ok=True)
    path = out / "eda_prematurity_alignment.png"
    fig.savefig(path, dpi=140)
    print(f"\n  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--arm", default="wide400")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "assets" / "eda")
    args = parser.parse_args()

    cache = load(args.data, args.arm)
    pre, local = interval_table(cache["position"], cache["record_id"])

    report(cache, pre, local)
    figure(cache, reanchor_absolute(cache, pre), reanchor_normalised(cache, pre, local), args.out)


if __name__ == "__main__":
    main()