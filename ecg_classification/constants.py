"""Dataset-level constants for the inter-patient MIT-BIH pipeline.

Everything that defines *which* records are used, *how* annotation symbols map
to AAMI classes, and *what shape* a beat has lives here. Nothing in this module
imports anything else from the project, so it can be read on its own as the
specification for the rebuilt pipeline.

Design notes
------------
The purpose of this rebuild is to change the *source* of the data from a
preprocessed CSV release to the raw PhysioNet records, not to reproduce that
release. The CSV's generation procedure is not documented, so reproducing it
is not possible and imitating it would mean inheriting undocumented choices.

Only the beat length of 187 samples is carried over, because the existing
network's input layer fixes it. Everything else -- sampling rate, window
placement, normalisation -- is specified here on physiological grounds and can
be defended without reference to the earlier artefact.
"""

from __future__ import annotations

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
#: the MLII selector in ``data.py`` even if they were kept.
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
#: MLII is the *second* channel rather than the first. ``data.py`` selects the
#: channel by name rather than by index, which handles this automatically; the
#: constant is kept for documentation and for assertions in tests.
REVERSED_CHANNEL_RECORDS: tuple[str, ...] = ("114",)


# ---------------------------------------------------------------------------
# AAMI class mapping
# ---------------------------------------------------------------------------

#: AAMI EC57 class labels, in a fixed index order. The model emits logits in
#: this order, so never reorder this tuple without retraining.
CLASS_NAMES: tuple[str, ...] = ("N", "S", "V", "F", "Q")

CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_INDEX.items()}

#: MIT-BIH beat annotation symbol -> AAMI class.
#:
#: Any symbol absent from this mapping is a non-beat annotation (rhythm change,
#: signal quality, artefact, and so on) and is dropped during segmentation.
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

#: IMPORTANT. Excluding the paced records removes essentially all "/" and "f"
#: beats, so after AAMI filtering the Q class is left with only a few dozen
#: genuinely unclassifiable beats across the whole database. Q is retained as
#: an output class so the network architecture is unchanged, but it is
#: effectively degenerate and per-class metrics for Q are not interpretable.
#:
#: This is also a further reason the previous CSV-based results are not
#: comparable: that release kept the paced records, so its Q class held roughly
#: eight thousand paced beats rather than a few dozen artefacts.
DEGENERATE_CLASSES: tuple[str, ...] = ("Q",)


# ---------------------------------------------------------------------------
# Target signal specification
# ---------------------------------------------------------------------------
#
# Only one property is inherited from the previous pipeline: a beat is 187
# samples long, so the network, the augmentation code and the training loop
# need no changes. Every other choice below is made here and is justified on
# physiological grounds rather than copied from an undocumented artefact.

#: Resampling target, in Hz. Chosen so that BEAT_LENGTH samples span roughly
#: one cardiac cycle: 187 / 250 = 0.748 s.
TARGET_FS = 250

#: Samples per beat. Fixed by the existing network's input layer.
BEAT_LENGTH = 187

#: Samples taken before the R peak. 62 / 250 = 0.248 s, which comfortably
#: contains the P wave (normally under 0.12 s) and the PR interval (0.12-0.20 s
#: normally, longer under first-degree block).
PRE_SAMPLES = 62

#: Samples taken from the R peak onward, inclusive. 125 / 250 = 0.500 s, which
#: contains the QRS complex (under 0.12 s normally, up to about 0.16 s for wide
#: complexes) and the T wave at physiological rates.
POST_SAMPLES = 125

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
    *(f"p_{c}" for c in CLASS_NAMES),
)

#: Class-by-record contingency table written at cache-build time. Feeds the
#: effective-patient-count analysis in P3.
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
    assert set(AAMI_SYMBOL_MAP.values()) <= set(CLASS_NAMES), "symbol map emits an unknown class"
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES)), "duplicate class name"
    assert PRE_SAMPLES + POST_SAMPLES == BEAT_LENGTH, (
        f"window halves sum to {PRE_SAMPLES + POST_SAMPLES}, expected {BEAT_LENGTH}"
    )


_validate()
