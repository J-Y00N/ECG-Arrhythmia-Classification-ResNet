# Changelog

## [Unreleased]

### Fixed
- Report and README overstated three results: the interventions (none improved on the baseline beyond the record-level interval, while inverse-frequency weighting collapsed training), the protocol gap (38% of the three-class macro F1, not half), and the design-effect residual, which had been attributed to unequal group sizes.
- Report: completed reference 11 and replaced an unchecked claim about the leads used by published classifiers with what their papers state.
- CHANGELOG: multi-label ranking metrics were listed as removed in 0.2.0; they remain in `metrics.py`.

## [0.2.0]

The pipeline was rebuilt from the raw PhysioNet records so that the recording is an explicit factor in the design. Results from 0.1.0 are not comparable and are excluded from every comparison; see `docs/report.md`, Appendix A.

### Added
- Raw WFDB loader (`mitdb.py`). Every beat keeps the recording it came from.
- Patient-disjoint protocol (`inter`, de Chazal DS1/DS2) alongside the beat-level one (`intra`), selected with `--protocol`.
- Five representation arms, selected with `ECG_REPRESENTATION`; the cache directory is derived from the arm.
- Optional normalised R–R interval features, joined at the classifier so the convolutional trunk is unchanged.
- Per-beat prediction table (`predictions.csv`) written by every run.
- `analysis/`: cluster bootstrap with intraclass correlation and design effect, per-record decomposition, calibration, capacity baselines, and the report's figures.
- `ECG_KEEP_Q=1` restores the five-class configuration.

### Changed
- Four model classes rather than five. After paced-record exclusion the unclassifiable class holds 15 artefact beats; they are kept as an observation set and never trained on.
- Preprocessing: 360 to 250 Hz polyphase resampling, R-peak-centred window, per-beat median subtraction and per-record robust scaling.
- MLII selected by channel name, which handles record 114.
- Model selection restricted to N, S and V, which every validation split can support.
- Class-weight exponent exposed, defaulting to 0.
- Learning curves plot the selection metric, macro F1 over N, S and V, rather than the unrestricted macro F1.

### Fixed
- Augmentation clipped its output to [0, 1], which saturated 9.5% of samples under the new normalisation and flattened the QRS complex.
- Time stretching moved the R peak by up to 19 samples.

### Removed
- CSV loader and the CSV-based figure generation.

## [0.1.0]

Residual 1D CNN trained on a preprocessed CSV release of MIT-BIH, beat-level split. Preserved at the tag `v0.1.0-csv`.