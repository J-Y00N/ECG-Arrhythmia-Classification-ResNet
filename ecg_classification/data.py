"""Build the MIT-BIH beat dataset from the raw PhysioNet records.

This module replaces the CSV loader. The public preprocessed release discards
record identifiers, which makes patient-level evaluation impossible; rebuilding
from the raw WFDB records restores the record identifier and therefore makes
the patient a usable unit of analysis rather than an invisible confounder.

The goal is to replace the data source, not to reproduce the CSV release. That
release's generation procedure is not documented anywhere we could verify, so
reproducing it is not possible and imitating it would mean inheriting choices
nobody can justify. The preprocessing below is therefore specified from first
principles; only the 187-sample beat length is inherited, because the existing
network's input layer fixes it.

Preprocessing decisions, and why
--------------------------------
1. R-peak locations come from the expert annotations, not from a detector.
   Detector error is removed as a confounder and the code gets simpler.
2. Beats are fixed windows centred on the R peak rather than windows whose
   length depends on the local median RR. The fixed window makes the
   representation independent of rhythm, so what the network sees is beat
   morphology alone.
3. Normalisation is per-beat median subtraction followed by a record-level
   robust scale. The first absorbs baseline wander without a filter; the second
   removes between-record gain differences while preserving amplitude contrast
   between beat types inside a record.
4. Paced records are excluded per AAMI EC57, so the Q class is close to empty.
5. Beats whose window runs past either end of the record are dropped rather
   than zero-padded, so no beat carries an artificial flat segment.

None of these affect the comparison of interest, because the intra-patient and
inter-patient protocols both run through this same pipeline.

Usage
-----
    python -m ecg_classification.data --build-cache
    python -m ecg_classification.data --build-cache --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .constants import (
    AAMI_RECORDS,
    AAMI_SYMBOL_MAP,
    BEAT_LENGTH,
    CLASS_NAMES,
    CLASS_RECORD_TABLE_NAME,
    CLASS_TO_INDEX,
    DS1,
    DS2,
    INTRA_TEST_FRACTION,
    MIN_SCALE,
    MITDB_NAME,
    POST_SAMPLES,
    PREFERRED_CHANNEL,
    PRE_SAMPLES,
    SOURCE_FS,
    STRICT_DISJOINT_EXCLUDE,
    TARGET_FS,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_mitdb(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch the MIT-BIH Arrhythmia Database into ``data_dir/mitdb``.

    Roughly 100 MB. Skipped if the directory already holds the header files for
    every AAMI record, so this is safe to call on every run.
    """
    import wfdb  # imported lazily so the module can be inspected without wfdb

    db_dir = data_dir / MITDB_NAME
    db_dir.mkdir(parents=True, exist_ok=True)

    missing = [r for r in AAMI_RECORDS if not (db_dir / f"{r}.hea").exists()]
    if not missing:
        LOGGER.info("mitdb already present at %s", db_dir)
        return db_dir

    LOGGER.info("downloading %d missing records into %s", len(missing), db_dir)
    wfdb.dl_database(MITDB_NAME, str(db_dir))
    return db_dir


# ---------------------------------------------------------------------------
# 2. Load one record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawRecord:
    """A single record after channel selection, before resampling."""

    record_id: str
    signal: np.ndarray        # (n_samples,) float64, at SOURCE_FS
    r_peaks: np.ndarray       # (n_beats,) int, sample index at SOURCE_FS
    labels: np.ndarray        # (n_beats,) int, AAMI class index
    fs: int
    channel_index: int


def load_record(record_id: str, db_dir: Path) -> RawRecord:
    """Load the MLII channel and the AAMI-mapped beat annotations.

    The channel is selected *by name*, not by index. Record 114 has its two
    signals reversed relative to every other record, and selecting by name
    handles that without a special case.
    """
    import wfdb

    stem = str(db_dir / record_id)
    record = wfdb.rdrecord(stem)

    if PREFERRED_CHANNEL not in record.sig_name:
        raise ValueError(
            f"record {record_id} has no {PREFERRED_CHANNEL} channel "
            f"(signals: {record.sig_name}). Paced records 102 and 104 lack it "
            f"and should already be excluded by AAMI filtering."
        )
    channel = record.sig_name.index(PREFERRED_CHANNEL)
    signal = np.asarray(record.p_signal[:, channel], dtype=np.float64)

    if record.fs != SOURCE_FS:
        raise ValueError(f"record {record_id} sampled at {record.fs} Hz, expected {SOURCE_FS}")

    annotation = wfdb.rdann(stem, "atr")
    keep = [
        (int(sample), CLASS_TO_INDEX[AAMI_SYMBOL_MAP[symbol]])
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in AAMI_SYMBOL_MAP
    ]
    if not keep:
        raise ValueError(f"record {record_id} produced no beat annotations")

    r_peaks = np.array([s for s, _ in keep], dtype=np.int64)
    labels = np.array([c for _, c in keep], dtype=np.int64)

    return RawRecord(
        record_id=record_id,
        signal=signal,
        r_peaks=r_peaks,
        labels=labels,
        fs=int(record.fs),
        channel_index=channel,
    )


# ---------------------------------------------------------------------------
# 3. Resample to the target rate
# ---------------------------------------------------------------------------

def resample_record(raw: RawRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample to ``TARGET_FS`` and rescale the annotation positions.

    ``resample_poly`` is used rather than naive interpolation because it applies
    an anti-aliasing filter. For 360 -> 250 Hz, gcd(250, 360) = 10, so the
    rational factor is 25/36.
    """
    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(TARGET_FS, SOURCE_FS)
    up, down = TARGET_FS // divisor, SOURCE_FS // divisor

    signal = resample_poly(raw.signal, up, down).astype(np.float64)

    scale = TARGET_FS / SOURCE_FS
    r_peaks = np.round(raw.r_peaks * scale).astype(np.int64)

    # A rounded peak can land one sample past the resampled signal.
    valid = (r_peaks >= 0) & (r_peaks < len(signal))
    return signal, r_peaks[valid], raw.labels[valid]


# ---------------------------------------------------------------------------
# 4. Segment into fixed-length beats
# ---------------------------------------------------------------------------

def segment_beats(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut fixed windows centred on the annotated R peaks.

    Each beat spans ``PRE_SAMPLES`` before the R peak and ``POST_SAMPLES`` from
    the R peak onward, for ``BEAT_LENGTH`` samples in total. Because the window
    does not depend on the local RR interval, the representation carries beat
    morphology and not rhythm.

    Normalisation runs in two stages. Each beat has its own median subtracted,
    which removes the local baseline offset and so absorbs slow baseline wander
    without a filter. The beat is then divided by a record-level robust scale,
    which removes between-record gain differences while preserving amplitude
    contrast between beat types inside a record. Those between-record
    differences are precisely the patient-level factors -- thoracic impedance,
    cardiac axis, recorder gain -- that this database cannot separate from one
    another, since it holds one recording per subject.

    Returns ``(beats, labels, positions)`` where ``positions`` holds the R-peak
    sample index of each beat, kept so a beat can be traced back to its place
    in the record.
    """
    empty = (
        np.empty((0, BEAT_LENGTH), dtype=np.float32),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )
    if signal.size < BEAT_LENGTH or r_peaks.size == 0:
        return empty

    # Drop beats whose window would overrun either end of the record. Padding
    # them instead would hand the network a flat segment that means nothing.
    keep = (r_peaks >= PRE_SAMPLES) & (r_peaks + POST_SAMPLES <= len(signal))
    peaks, beat_labels = r_peaks[keep], labels[keep]
    if peaks.size == 0:
        return empty

    offsets = np.arange(-PRE_SAMPLES, POST_SAMPLES, dtype=np.int64)
    windows = signal[peaks[:, None] + offsets[None, :]]

    windows = windows - np.median(windows, axis=1, keepdims=True)
    windows = windows / _record_scale(signal)

    return (
        windows.astype(np.float32),
        beat_labels.astype(np.int64),
        peaks.astype(np.int64),
    )


def _record_scale(signal: np.ndarray) -> float:
    """Robust amplitude scale for one record: the interquartile range.

    The IQR is preferred to the standard deviation because Holter recordings
    carry motion artefacts and electrode transients whose extreme values would
    dominate a variance-based estimate.
    """
    high, low = np.percentile(signal, [75, 25])
    scale = float(high - low)
    return scale if scale > MIN_SCALE else 1.0


# ---------------------------------------------------------------------------
# 5. Cache
# ---------------------------------------------------------------------------

def build_cache(
    data_dir: Path = DEFAULT_DATA_DIR,
    records: tuple[str, ...] = AAMI_RECORDS,
    force: bool = False,
) -> Path:
    """Segment every record once and cache the result as a single ``.npz``.

    The cache carries a ``record_id`` for every beat. Everything downstream in
    P3 depends on that column existing; without it the cluster bootstrap, the
    per-record decomposition and the effective patient count cannot be computed
    without re-running inference.

    Also writes the class-by-record contingency table, which feeds the
    effective-patient-count analysis.
    """
    db_dir = download_mitdb(data_dir)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "beats.npz"

    if cache_path.exists() and not force:
        LOGGER.info("cache already built at %s (pass --force to rebuild)", cache_path)
        return cache_path

    all_beats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_records: list[np.ndarray] = []
    all_positions: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}

    for record_id in records:
        raw = load_record(record_id, db_dir)
        signal, r_peaks, labels = resample_record(raw)
        beats, beat_labels, positions = segment_beats(signal, r_peaks, labels)

        all_beats.append(beats)
        all_labels.append(beat_labels)
        all_positions.append(positions)
        all_records.append(np.full(len(beats), record_id, dtype="<U3"))

        per_class = {name: 0 for name in CLASS_NAMES}
        for index, count in zip(*np.unique(beat_labels, return_counts=True)):
            per_class[CLASS_NAMES[int(index)]] = int(count)
        counts[record_id] = per_class

        LOGGER.info(
            "%s: %6d beats  channel=%d  %s",
            record_id, len(beats), raw.channel_index,
            " ".join(f"{k}={v}" for k, v in per_class.items() if v),
        )

    x = np.concatenate(all_beats)
    y = np.concatenate(all_labels)
    record_ids = np.concatenate(all_records)
    positions = np.concatenate(all_positions)

    np.savez_compressed(
        cache_path, x=x, y=y, record_id=record_ids, position=positions
    )
    LOGGER.info("wrote %s  (%d beats, %.1f MB)", cache_path, len(x), cache_path.stat().st_size / 1e6)

    _write_class_record_table(cache_dir / CLASS_RECORD_TABLE_NAME, counts)
    _write_manifest(cache_dir / "cache_manifest.json", records, counts)
    return cache_path


def _write_class_record_table(path: Path, counts: dict[str, dict[str, int]]) -> None:
    """Write the class-by-record contingency table used by the P3 analysis."""
    header = "record_id," + ",".join(CLASS_NAMES) + ",total\n"
    lines = [header]
    for record_id in sorted(counts):
        row = counts[record_id]
        values = [row[name] for name in CLASS_NAMES]
        lines.append(f"{record_id}," + ",".join(map(str, values)) + f",{sum(values)}\n")
    path.write_text("".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", path)


def _write_manifest(path: Path, records: tuple[str, ...], counts: dict[str, dict[str, int]]) -> None:
    """Record the exact preprocessing parameters alongside the cache."""
    manifest = {
        "records": list(records),
        "n_records": len(records),
        "target_fs": TARGET_FS,
        "source_fs": SOURCE_FS,
        "beat_length": BEAT_LENGTH,
        "pre_samples": PRE_SAMPLES,
        "post_samples": POST_SAMPLES,
        "pre_seconds": round(PRE_SAMPLES / TARGET_FS, 4),
        "post_seconds": round(POST_SAMPLES / TARGET_FS, 4),
        "normalisation": "per-beat median subtraction, per-record IQR scale",
        "channel": PREFERRED_CHANNEL,
        "class_names": list(CLASS_NAMES),
        "total_beats": sum(sum(v.values()) for v in counts.values()),
        "class_totals": {
            name: sum(v[name] for v in counts.values()) for name in CLASS_NAMES
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", path)


def load_cache(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, np.ndarray]:
    """Load the cached arrays. Keys: ``x``, ``y``, ``record_id``, ``position``."""
    cache_path = data_dir / "cache" / "beats.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"no cache at {cache_path}. Run: python -m ecg_classification.data --build-cache"
        )
    with np.load(cache_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("x", "y", "record_id", "position")}


# ---------------------------------------------------------------------------
# 6. Protocol splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """Index arrays into the cached beat table, plus a description of how they
    were produced. ``asdict`` output is written into ``config.json`` so the
    protocol used by a run is recoverable from the artefacts alone."""

    protocol: str
    seed: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_records: tuple[str, ...]
    test_records: tuple[str, ...]
    strict_disjoint: bool

    def describe(self) -> dict:
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in {"train_index", "test_index"}
        }
        payload["n_train"] = int(len(self.train_index))
        payload["n_test"] = int(len(self.test_index))
        return payload


def make_split(
    record_ids: np.ndarray,
    labels: np.ndarray,
    protocol: str,
    seed: int = 42,
    strict_disjoint: bool = False,
) -> Split:
    """Build train and test index arrays for one protocol.

    ``inter`` assigns whole records to the two halves following de Chazal.
    ``intra`` pools every beat and splits at the beat level, which reproduces
    the split style of the public CSV release inside this pipeline. Because
    both protocols consume identical preprocessing, any difference in results
    is attributable to the split alone.

    ``strict_disjoint`` additionally drops record 202, whose subject also
    appears in DS1 as record 201.
    """
    if protocol not in {"intra", "inter"}:
        raise ValueError(f"unknown protocol {protocol!r}")

    excluded = set(STRICT_DISJOINT_EXCLUDE) if strict_disjoint else set()

    if protocol == "inter":
        train_records = tuple(r for r in DS1 if r not in excluded)
        test_records = tuple(r for r in DS2 if r not in excluded)
        train_index = np.where(np.isin(record_ids, train_records))[0]
        test_index = np.where(np.isin(record_ids, test_records))[0]
    else:
        usable = tuple(r for r in AAMI_RECORDS if r not in excluded)
        pool = np.where(np.isin(record_ids, usable))[0]
        train_index, test_index = _stratified_beat_split(
            pool, labels[pool], INTRA_TEST_FRACTION, seed
        )
        train_records = test_records = usable

    return Split(
        protocol=protocol,
        seed=seed,
        train_index=train_index,
        test_index=test_index,
        train_records=train_records,
        test_records=test_records,
        strict_disjoint=strict_disjoint,
    )


def _stratified_beat_split(
    pool: np.ndarray, labels: np.ndarray, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified beat-level split, deliberately ignoring record identity."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_index in np.unique(labels):
        members = pool[labels == class_index]
        shuffled = rng.permutation(members)
        n_test = int(round(len(shuffled) * test_fraction))
        test_parts.append(shuffled[:n_test])
        train_parts.append(shuffled[n_test:])

    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(test_parts)),
    )


# ---------------------------------------------------------------------------
# 7. Torch dataset
# ---------------------------------------------------------------------------

class ECGBeatDataset:
    """Minimal dataset over the cached beats.

    Yields ``(signal, label, record_id, beat_index)``. The last two are carried
    through so the evaluation loop can write the prediction table without
    re-deriving provenance. Adjust the return signature to match the existing
    ``train.py`` collate expectations, but keep ``record_id`` reaching the
    prediction table.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        record_ids: np.ndarray,
        index: np.ndarray,
        transform=None,
    ) -> None:
        self.x = x
        self.y = y
        self.record_ids = record_ids
        self.index = index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int):
        i = int(self.index[position])
        signal = self.x[i]
        if self.transform is not None:
            signal = self.transform(signal)
        signal = np.ascontiguousarray(signal, dtype=np.float32)[None, :]  # (1, 187)
        return signal, int(self.y[i]), str(self.record_ids[i]), i

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        values, found = np.unique(self.y[self.index], return_counts=True)
        counts[values] = found
        return counts


def build_datasets(
    protocol: str,
    seed: int = 42,
    data_dir: Path = DEFAULT_DATA_DIR,
    strict_disjoint: bool = False,
    train_transform=None,
) -> tuple[ECGBeatDataset, ECGBeatDataset, Split]:
    """Convenience entry point for ``train.py``."""
    cache = load_cache(data_dir)
    split = make_split(
        cache["record_id"], cache["y"], protocol, seed, strict_disjoint
    )
    train = ECGBeatDataset(
        cache["x"], cache["y"], cache["record_id"], split.train_index, train_transform
    )
    test = ECGBeatDataset(
        cache["x"], cache["y"], cache["record_id"], split.test_index, None
    )
    return train, test, split


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild an existing cache")
    parser.add_argument("--summary", action="store_true", help="print split sizes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.build_cache:
        build_cache(args.data_dir, force=args.force)

    if args.summary:
        cache = load_cache(args.data_dir)
        for protocol in ("intra", "inter"):
            split = make_split(cache["record_id"], cache["y"], protocol, seed=42)
            print(json.dumps(split.describe(), indent=2, default=list))


if __name__ == "__main__":
    main()
