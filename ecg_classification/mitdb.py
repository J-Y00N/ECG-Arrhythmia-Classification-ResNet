"""Build the MIT-BIH beat cache from the raw PhysioNet records.

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
6. Where the extraction window is longer than the model input, it is resampled
   onto it. Which window applies is chosen by the representation arm; see
   ``constants``.

None of these affect the comparison of interest, because the intra-patient and
inter-patient protocols both run through this same pipeline.

Usage
-----
    python -m ecg_classification.mitdb --build-cache
    python -m ecg_classification.mitdb --build-cache --data-dir /path/to/data

This module owns the raw-to-cache layer only. Turning the cache into torch
Datasets, applying augmentation and building samplers stays in ``data.py``.
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
    AAMI_SYMBOLS,
    AAMI_SYMBOL_MAP,
    AAMI_SYMBOL_TO_INDEX,
    BEAT_LENGTH,
    CLASS_RECORD_TABLE_NAME,
    DS1,
    DS2,
    EFFECTIVE_FS,
    INTRA_TEST_FRACTION,
    MIN_SCALE,
    MITDB_NAME,
    NEEDS_RESAMPLE,
    POST_SAMPLES,
    PREFERRED_CHANNEL,
    PRE_SAMPLES,
    PROTOCOLS,
    REPRESENTATION,
    SAMPLE_LENGTH,
    SOURCE_FS,
    STRICT_DISJOINT_EXCLUDE,
    TARGET_FS,
    WINDOW_LENGTH,
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
        (int(sample), AAMI_SYMBOL_TO_INDEX[AAMI_SYMBOL_MAP[symbol]])
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
    the R peak onward, for ``WINDOW_LENGTH`` samples in total. Because the window
    does not depend on the local RR interval, any rhythm information it carries
    comes from neighbouring beats falling inside it rather than from the window
    being scaled to the local rate.

    Where ``WINDOW_LENGTH`` exceeds the model input length the window is
    resampled onto it, so that every representation arm hands the network an
    identically shaped input and a difference in results cannot be attributed to
    the architecture seeing a different number of samples.

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
    if signal.size < WINDOW_LENGTH or r_peaks.size == 0:
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

    if NEEDS_RESAMPLE:
        # Polyphase rather than interpolation. Lowering the effective rate puts
        # the Nyquist limit below components that are present in the signal, and
        # plain interpolation would fold them back as low-frequency noise rather
        # than removing them. resample_poly applies the anti-aliasing filter that
        # makes the loss a clean one.
        from scipy.signal import resample_poly

        windows = resample_poly(windows, SAMPLE_LENGTH, WINDOW_LENGTH, axis=1)

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

    LOGGER.info(
        "representation=%s  window=%d (%.3f/%.3f s)  input=%d  effective_fs=%.1f Hz",
        REPRESENTATION, WINDOW_LENGTH, PRE_SAMPLES / TARGET_FS,
        POST_SAMPLES / TARGET_FS, BEAT_LENGTH, EFFECTIVE_FS,
    )

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

        per_class = {symbol: 0 for symbol in AAMI_SYMBOLS}
        for index, count in zip(*np.unique(beat_labels, return_counts=True)):
            per_class[AAMI_SYMBOLS[int(index)]] = int(count)
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
    header = "record_id," + ",".join(AAMI_SYMBOLS) + ",total\n"
    lines = [header]
    for record_id in sorted(counts):
        row = counts[record_id]
        values = [row[symbol] for symbol in AAMI_SYMBOLS]
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
        "representation": REPRESENTATION,
        "window_length": WINDOW_LENGTH,
        "model_input_length": BEAT_LENGTH,
        "resampled": NEEDS_RESAMPLE,
        "effective_fs": EFFECTIVE_FS,
        "pre_samples": PRE_SAMPLES,
        "post_samples": POST_SAMPLES,
        "pre_seconds": round(PRE_SAMPLES / TARGET_FS, 4),
        "post_seconds": round(POST_SAMPLES / TARGET_FS, 4),
        "normalisation": "per-beat median subtraction, per-record IQR scale",
        "channel": PREFERRED_CHANNEL,
        "aami_symbols": list(AAMI_SYMBOLS),
        "total_beats": sum(sum(v.values()) for v in counts.values()),
        "class_totals": {
            symbol: sum(v[symbol] for v in counts.values()) for symbol in AAMI_SYMBOLS
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", path)


def load_cache(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, np.ndarray]:
    """Load the cached arrays. Keys: ``x``, ``y``, ``record_id``, ``position``."""
    cache_path = data_dir / "cache" / "beats.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"no cache at {cache_path}. Run: python -m ecg_classification.mitdb --build-cache"
        )
    with np.load(cache_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("x", "y", "record_id", "position")}


# ---------------------------------------------------------------------------
# 6. Protocol splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """Index arrays into the cached beat table, plus a description of how they
    were produced. ``describe()`` output is written into ``config.json`` so the
    protocol used by a run is recoverable from the artefacts alone."""

    protocol: str
    seed: int
    train_index: np.ndarray
    valid_index: np.ndarray
    test_index: np.ndarray
    train_records: tuple[str, ...]
    valid_records: tuple[str, ...]
    test_records: tuple[str, ...]
    strict_disjoint: bool

    def describe(self) -> dict:
        payload = {
            key: value for key, value in asdict(self).items()
            if not key.endswith("_index")
        }
        payload["n_train"] = int(len(self.train_index))
        payload["n_valid"] = int(len(self.valid_index))
        payload["n_test"] = int(len(self.test_index))
        return payload


def make_split(
    record_ids: np.ndarray,
    labels: np.ndarray,
    protocol: str,
    seed: int = 42,
    validation_size: float = 0.10,
    strict_disjoint: bool = False,
) -> Split:
    """Build train, validation and test index arrays for one protocol.

    ``inter`` assigns whole records following de Chazal: DS2 is the test half,
    and DS1 is divided into training and validation records. The validation
    split is made at the record level too, because holding out beats from
    records the model also trains on would reintroduce exactly the leakage the
    protocol exists to remove -- early stopping would then be tuned against a
    contaminated signal.

    ``intra`` pools every beat and splits at the beat level, reproducing the
    split style of the public CSV release inside this pipeline. Because both
    protocols consume identical preprocessing, any difference in results is
    attributable to the split alone.

    ``strict_disjoint`` additionally drops record 202, whose subject also
    appears in DS1 as record 201.

    Caveat on inter-patient validation. Minority classes are concentrated in a
    handful of records, so a validation half of two or three records may carry
    almost no F or Q beats. Validation macro F1 is therefore noisy for those
    classes, and model selection should be read as being driven by N, S and V.
    Running several seeds varies the validation records as well as the
    initialisation, which makes that instability visible rather than hidden.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {PROTOCOLS}")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must lie in (0, 1)")

    excluded = set(STRICT_DISJOINT_EXCLUDE) if strict_disjoint else set()
    rng = np.random.default_rng(seed)

    if protocol == "inter":
        pool = [r for r in DS1 if r not in excluded]
        n_valid = max(1, int(round(len(pool) * validation_size)))
        shuffled = rng.permutation(np.asarray(pool))
        valid_records = tuple(sorted(shuffled[:n_valid].tolist()))
        train_records = tuple(sorted(shuffled[n_valid:].tolist()))
        test_records = tuple(r for r in DS2 if r not in excluded)

        train_index = np.where(np.isin(record_ids, train_records))[0]
        valid_index = np.where(np.isin(record_ids, valid_records))[0]
        test_index = np.where(np.isin(record_ids, test_records))[0]
    else:
        usable = tuple(r for r in AAMI_RECORDS if r not in excluded)
        pool_index = np.where(np.isin(record_ids, usable))[0]
        train_index, valid_index, test_index = _stratified_beat_split(
            pool_index, labels[pool_index], validation_size, INTRA_TEST_FRACTION, rng
        )
        train_records = valid_records = test_records = usable

    return Split(
        protocol=protocol,
        seed=seed,
        train_index=train_index,
        valid_index=valid_index,
        test_index=test_index,
        train_records=train_records,
        valid_records=valid_records,
        test_records=test_records,
        strict_disjoint=strict_disjoint,
    )


def _stratified_beat_split(
    pool: np.ndarray,
    labels: np.ndarray,
    validation_size: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified beat-level split, deliberately ignoring record identity.

    ``validation_size`` is taken as a fraction of the training remainder, which
    matches how the previous pipeline carved out its validation set.
    """
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_index in np.unique(labels):
        members = rng.permutation(pool[labels == class_index])
        n_test = int(round(len(members) * test_fraction))
        test_parts.append(members[:n_test])

        remainder = members[n_test:]
        n_valid = int(round(len(remainder) * validation_size))
        valid_parts.append(remainder[:n_valid])
        train_parts.append(remainder[n_valid:])

    return (
        np.sort(np.concatenate(train_parts)),
        np.sort(np.concatenate(valid_parts)),
        np.sort(np.concatenate(test_parts)),
    )


# ---------------------------------------------------------------------------
# 7. CLI
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