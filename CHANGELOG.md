# Changelog

## [Unreleased]

### Added
- `docs/poster/`: the one-page poster as PDF with a PNG preview, linked from the README. Its text and charts match the report after the corrections below.

### Fixed
- Report: Table 11 gave the four-class macro F1 of `wide400` as 0.6070; the artefacts give 0.6073.
- `wide187_rr` was missing from the representation table, so `ECG_REPRESENTATION=wide187_rr` raised although the arm is documented and its runs are stored. Restored.
- Run directories for the default narrow arm are named without an arm prefix again (`inter-none-lossweight0-seed42`), matching the stored runs, the README, the notebook and `analysis/`.
- `class_weights` now warns when a class falls below `min_support` and receives zero weight. This happens in every seed-42 inter-patient run, where fusion keeps 42 training beats; the report now says so.
- `analysis.bootstrap` reports the accuracy row of Table 7, the design-effect check, which no committed code produced.
- Baseline `config.json` records the 1-NN reference subsample and that validation beats are folded into training.
- `analysis.baselines` uses the full 1-NN reference set by default. Table 10 is rerun on it: 1-NN reaches 0.948 intra and 0.513 inter (gap 0.435, correlation 0.358), against 0.918, 0.522, 0.396 and 0.418 with the earlier 20,000-beat subsample.
- The accuracy bootstrap in `analysis.bootstrap` draws from its own generator, so adding it leaves the paired comparisons printed after it unchanged.
- Report: Table 12 is rerun on the split every other table uses, with paired intervals. No exponent improves on zero beyond the record-level interval; $\beta = 0.75$ scores highest (+0.086) with an interval containing zero. The collapse at $\beta = 1$ occurs only on the earlier split that kept fusion in training, and is now described as such.
- Report: prematurity ratios (0.754 and 0.710, fusion 0.985) and the amplitude range and share outside [0, 1] behind the clipping defect are taken from the current caches.
- Report: disclosed the fusion zero weight, the training settings (AdamW, label smoothing 0.05, LR schedule), the 18/4 record split and the baselines' differences from the network; labelled the §4.2 and Table 7 figures as `wide187`; noted that Table 12 comes from a different validation split; qualified the global-pooling claim; corrected the seed-versus-record comparison (about twelve times, not forty, and seeds also change the validation split), the number of underconfident recordings (17, not 20), recordings above 0.99 accuracy (8, not 6), the largest group size (3,247), the logistic ICC (0.298), the `wide400` seed count and the training-recording ratio.
- Figure 3, the notebook and `baselines.py` still said the gap was about half the score.
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
- Augmentation clipped its output to [0, 1], which altered 62% of samples under the new normalisation and flattened the QRS complex.
- Time stretching moved the R peak by up to 19 samples.

### Removed
- CSV loader and the CSV-based figure generation.

## [0.1.0]

Residual 1D CNN trained on a preprocessed CSV release of MIT-BIH, beat-level split. Preserved at the tag `v0.1.0-csv`.