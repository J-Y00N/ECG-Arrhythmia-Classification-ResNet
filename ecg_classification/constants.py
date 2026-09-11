"""Dataset-level constants for the inter-patient MIT-BIH pipeline.

Everything that defines *which* records are used, *how* annotation symbols map
to AAMI classes, and *what shape* a beat has lives here. Nothing in this module
imports anything else from the project, so it can be read on its own as the
specification for the pipeline.

Design notes
------------
The purpose of the rebuild is to change the *source* of the data from a
preprocessed CSV release to the raw PhysioNet records, not to reproduce that
release. The CSV's generation procedure is not documented, so reproducing it is
not possible and imitating it would mean inheriting undocumented choices.

Only the model input length of 187 samples is carried over, so that the network
sees an identically shaped input in every arm. Everything else -- sampling rate,
window placement, normalisation -- is specified here on physiological grounds and
can be defended without reference to the earlier artefact.

Representation arms
-------------------
How much signal the window covers is a factor of the experiment, selected by the
``ECG_REPRESENTATION`` environment variable rather than by editing this file.
Twenty runs edited by hand is a mistake waiting to happen, and the arm is written
into every run's configuration so that it cannot be lost after the fact.

Naming
------
``CLASS_SYMBOLS`` holds the compact AAMI codes ("N", "S", "V", "F", "Q") and is
what the pipeline uses for column headers, file names and lookups.
``CLASS_NAMES`` holds the long human-readable descriptions and is what figures
and classification reports display. They are different things; do not conflate
them.
"""

from __future__ import annotations

import os as _os

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
#
# Two levels of label live here and must not be conflated.
#
# The *database* level is the full AAMI five-class grouping. Segmentation needs
# it, because every annotated beat has to be assigned somewhere before any
# modelling decision is taken, and the class-by-record table describes what the
# database contains rather than what we chose to learn.
#
# The *model* level is four classes. Once the paced recordings are excluded per
# AAMI EC57, the Q class holds 15 unclassifiable beats database-wide -- all of
# them baseline wander or electrode transients. de Chazal et al. (2004) dropped
# Q for this reason and the inter-patient literature has followed since, so a
# four-class head is both the honest choice and the one that keeps results
# comparable with published work.
#
# Indices 0-3 mean the same thing at both levels, so a cache built with the
# five-class mapping stays valid without rebuilding.

#: Full AAMI grouping. Database level: used for segmentation and for the
#: class-by-record contingency table.
AAMI_LABEL_TO_SYMBOL = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}

AAMI_SYMBOLS = [AAMI_LABEL_TO_SYMBOL[index] for index in sorted(AAMI_LABEL_TO_SYMBOL)]

AAMI_SYMBOL_TO_INDEX: dict[str, int] = {
    symbol: index for index, symbol in AAMI_LABEL_TO_SYMBOL.items()
}

#: Label the model does not learn. Beats carrying it are held out of every
#: split and kept only as an observation set: they let us ask how a classifier
#: that has never seen pure artefact behaves when it meets some. A confident
#: "normal beat" on a detached-electrode waveform is a clinically dangerous
#: failure mode, and 15 beats is enough to look at even though it is nowhere
#: near enough to score.
OBSERVATION_LABEL = 4
OBSERVATION_SYMBOL = AAMI_LABEL_TO_SYMBOL[OBSERVATION_LABEL]

#: Model level. Compact codes, in fixed index order. The network emits logits in
#: this order, so never reorder without retraining.
CLASS_SYMBOLS = [
    symbol for index, symbol in sorted(AAMI_LABEL_TO_SYMBOL.items())
    if index != OBSERVATION_LABEL
]

LABEL_TO_NAME = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
    3: "Fusion beat",
}

#: Long descriptions, used for figure labels and classification reports.
CLASS_NAMES = [LABEL_TO_NAME[index] for index in sorted(LABEL_TO_NAME)]

NUM_CLASSES = len(CLASS_NAMES)

#: Kept for backwards compatibility with code that indexed symbols by label.
LABEL_TO_SYMBOL = {
    index: symbol for index, symbol in AAMI_LABEL_TO_SYMBOL.items()
    if index != OBSERVATION_LABEL
}

SYMBOL_TO_INDEX: dict[str, int] = {
    symbol: index for index, symbol in LABEL_TO_SYMBOL.items()
}

#: Samples per beat. Fixed by the existing network's input layer.
SAMPLE_LENGTH = 187

#: Alias used by the raw-data pipeline. Same value, clearer name in context.
BEAT_LENGTH = SAMPLE_LENGTH


# ---------------------------------------------------------------------------
# Source database
# ---------------------------------------------------------------------------

MITDB_NAME = "mitdb"

#: Native sampling rate of the MIT-BIH Arrhythmia Database, in Hz.
SOURCE_FS = 360


# ---------------------------------------------------------------------------
# Record partitions
# ---------------------------------------------------------------------------

#: Records excluded under the AAMI EC57 recommendation because they contain
#: paced beats. Records 102 and 104 additionally have no MLII channel at all
#: (surgical dressings forced the use of V5), so they could not be loaded by
#: the MLII selector in ``mitdb.py`` even if they were kept.
PACED_RECORDS: tuple[str, ...] = ("102", "104", "107", "217")

#: de Chazal et al. (2004) inter-patient partition, training half.
DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
)

#: de Chazal et al. (2004) inter-patient partition, evaluation half.
DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
)

#: The 44 records that remain after AAMI paced-record exclusion.
AAMI_RECORDS: tuple[str, ...] = tuple(sorted(DS1 + DS2))

#: Records 201 and 202 come from the same male subject, and the de Chazal
#: partition places 201 in DS1 and 202 in DS2. The published "inter-patient"
#: split is therefore not strictly subject-disjoint. The effect is expected to
#: be small (one subject out of 47), but it is a real caveat and worth a
#: sensitivity run.
SAME_SUBJECT_PAIRS: tuple[tuple[str, str], ...] = (("201", "202"),)

#: Records to drop for the strict subject-disjoint sensitivity analysis.
STRICT_DISJOINT_EXCLUDE: tuple[str, ...] = ("202",)

#: Record 114 has its two signals reversed relative to every other record, so
#: MLII is the *second* channel rather than the first. ``mitdb.py`` selects the
#: channel by name rather than by index, which handles this automatically; the
#: constant is kept for documentation and for assertions in tests.
REVERSED_CHANNEL_RECORDS: tuple[str, ...] = ("114",)


# ---------------------------------------------------------------------------
# AAMI symbol mapping
# ---------------------------------------------------------------------------

#: MIT-BIH beat annotation symbol -> AAMI class symbol.
#:
#: Any symbol absent from this mapping is a non-beat annotation (rhythm change,
#: signal quality, artefact, and so on) and is dropped during segmentation.
#:
#: "/" and "f" survive here for completeness, but they occur only in the paced
#: recordings, which AAMI excludes. What remains under Q after that exclusion is
#: the literal "Q" annotation: 15 beats the original annotators could not
#: classify. This is also why the previous CSV-based results are not comparable
#: with these -- that release kept the paced recordings, so its Q class was
#: roughly eight thousand paced beats, a visually distinctive and easily learned
#: category making up about 7% of the data, rather than a handful of artefacts.
AAMI_SYMBOL_MAP: dict[str, str] = {
    # N: normal, bundle-branch block, nodal escape, atrial escape
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S: supraventricular ectopic
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V: ventricular ectopic
    "V": "V",
    "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
    # Q: unclassifiable, paced, fusion of paced and normal
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


# ---------------------------------------------------------------------------
# Target signal specification
# ---------------------------------------------------------------------------
#
# Two lengths live here and must not be conflated.
#
# The *window* is how much signal is cut out of the recording, measured in
# samples at TARGET_FS. It is a physiological choice: how much of the cardiac
# cycle the network is allowed to see.
#
# The *model input* is the length the network actually receives. It is held at
# 187 across the arms so that the architecture's relationship to its input never
# changes, which is what lets a difference in results be attributed to the data
# rather than to the model.
#
# When the two differ the window is resampled onto the input length, trading
# temporal resolution for context. That trade is the point of the wide arms, not
# a side effect of them.

#: Resampling target, in Hz.
TARGET_FS = 250

#: Representation arms.
#:
#: ``narrow`` takes 0.248 s before the R peak, enough for the P wave (normally
#: under 0.12 s) and the PR interval (0.12-0.20 s, longer under first-degree
#: block), and 0.500 s after it, enough for the QRS complex and the T wave. It
#: contains exactly one beat and therefore carries morphology without rhythm.
#:
#: ``wide187`` reaches 0.80 s either side, at which point the previous R peak is
#: visible for 98.6% of supraventricular beats against 55.5% of normal ones --
#: the widest separation available -- while beats two cycles back stay under 2%.
#: Resampling 400 samples onto 187 puts the effective rate at 117 Hz, so the QRS
#: complex falls from roughly 22 samples to 10 and components above 58.5 Hz are
#: filtered away before they can fold back.
#:
#: ``wide400`` holds the same window at full rate. Its only purpose is to say
#: whether any loss seen in ``wide187`` came from the resampling or from the
#: wider context diluting the complex.
_REPRESENTATIONS: dict[str, dict[str, int]] = {
    "narrow":  {"pre": 62,  "post": 125, "input": 187},
    "wide187": {"pre": 200, "post": 200, "input": 187},
    "wide400": {"pre": 200, "post": 200, "input": 400},
}

REPRESENTATION = _os.environ.get("ECG_REPRESENTATION", "narrow")
if REPRESENTATION not in _REPRESENTATIONS:
    raise ValueError(
        f"unknown ECG_REPRESENTATION {REPRESENTATION!r}; "
        f"expected one of {sorted(_REPRESENTATIONS)}"
    )

_arm = _REPRESENTATIONS[REPRESENTATION]

#: Samples taken before and after the R peak, at TARGET_FS, before resampling.
PRE_SAMPLES: int = _arm["pre"]
POST_SAMPLES: int = _arm["post"]

#: Length of the extracted window, before resampling.
WINDOW_LENGTH: int = PRE_SAMPLES + POST_SAMPLES

#: Model input length. Equal to WINDOW_LENGTH when no resampling is needed.
SAMPLE_LENGTH: int = _arm["input"]

#: Alias used by the raw-data pipeline. Same value, clearer name in context.
BEAT_LENGTH = SAMPLE_LENGTH

#: Whether segmentation has to resample the window onto the input length.
NEEDS_RESAMPLE: bool = WINDOW_LENGTH != SAMPLE_LENGTH

#: Where the R peak sits in the model input, after any resampling. Equal to
#: PRE_SAMPLES when none happens. The augmenter needs it to restore alignment
#: after a time stretch, and the EDA figures mark it.
R_PEAK_INDEX: int = round(PRE_SAMPLES * SAMPLE_LENGTH / WINDOW_LENGTH)

#: Window reach either side of the R peak, in seconds. Reported rather than
#: computed downstream so that a run's configuration records it directly.
PRE_SECONDS: float = round(PRE_SAMPLES / TARGET_FS, 4)
POST_SECONDS: float = round(POST_SAMPLES / TARGET_FS, 4)

#: Effective sampling rate of the model input, in Hz.
EFFECTIVE_FS: float = round(SAMPLE_LENGTH / (WINDOW_LENGTH / TARGET_FS), 1)

#: Channel to extract. Present in all 44 AAMI records.
PREFERRED_CHANNEL = "MLII"

#: Floor for the record-level amplitude scale, guarding against flat records.
MIN_SCALE = 1e-8


# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------

#: ``inter`` uses the de Chazal record partition. ``intra`` pools every beat
#: from all 44 records and splits at the beat level, reproducing the split
#: style of the public CSV release inside this pipeline so that the two can be
#: compared without any other difference.
PROTOCOLS: tuple[str, ...] = ("intra", "inter")

#: Test fraction for the intra-patient protocol. 0.20 matches the public
#: release's 87,554 / 21,892 train-test ratio.
INTRA_TEST_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

#: Per-beat prediction table written at evaluation time.
#:
#: ``record_id`` is the single most important column in this project: without
#: it the cluster bootstrap, the per-record decomposition and the effective
#: patient count are all impossible to compute after the fact.
PREDICTION_COLUMNS: tuple[str, ...] = (
    "record_id",
    "beat_index",
    "y_true",
    "y_pred",
    *(f"p_{symbol}" for symbol in CLASS_SYMBOLS),
)

#: Class-by-record contingency table written at cache-build time. Feeds the
#: effective-patient-count analysis.
CLASS_RECORD_TABLE_NAME = "class_record_counts.csv"


# ---------------------------------------------------------------------------
# Integrity checks (run at import; cheap and catch typos in the record lists)
# ---------------------------------------------------------------------------

def _validate() -> None:
    assert len(DS1) == 22, f"DS1 should hold 22 records, found {len(DS1)}"
    assert len(DS2) == 22, f"DS2 should hold 22 records, found {len(DS2)}"
    assert not set(DS1) & set(DS2), "DS1 and DS2 overlap"
    assert len(AAMI_RECORDS) == 44, f"expected 44 AAMI records, found {len(AAMI_RECORDS)}"
    assert not set(AAMI_RECORDS) & set(PACED_RECORDS), "a paced record leaked into the AAMI set"
    assert len(CLASS_SYMBOLS) == len(CLASS_NAMES) == NUM_CLASSES == 4, "class label tables disagree"
    assert len(set(AAMI_SYMBOLS)) == 5, "the AAMI grouping should hold five classes"
    assert set(AAMI_SYMBOL_MAP.values()) <= set(AAMI_SYMBOLS), "symbol map emits an unknown class"
    assert set(SYMBOL_TO_INDEX) == set(CLASS_SYMBOLS), "symbol index table disagrees"
    assert OBSERVATION_SYMBOL not in CLASS_SYMBOLS, "the observation class must not be a model class"
    assert AAMI_SYMBOLS[:NUM_CLASSES] == CLASS_SYMBOLS, (
        "model indices must match AAMI indices 0..3 so an existing cache stays valid"
    )
    assert WINDOW_LENGTH == PRE_SAMPLES + POST_SAMPLES, "window halves disagree"
    assert BEAT_LENGTH == SAMPLE_LENGTH, "input length aliases disagree"
    assert 0 < R_PEAK_INDEX < SAMPLE_LENGTH, (
        f"R peak lands at index {R_PEAK_INDEX}, outside a model input of "
        f"{SAMPLE_LENGTH} samples"
    )
    assert NEEDS_RESAMPLE == (WINDOW_LENGTH != SAMPLE_LENGTH), "resample flag disagrees"


_validate()