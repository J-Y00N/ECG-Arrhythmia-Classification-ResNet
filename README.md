# ECG Arrhythmia Classification: Measuring the Inter-Patient Penalty

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH-0A7E8C)
![Device](https://img.shields.io/badge/Device-MPS%20%7C%20CUDA%20%7C%20CPU-444444)

A 1D residual CNN on MIT-BIH, rebuilt from the raw PhysioNet records so that the **recording** is an explicit factor in the design rather than an invisible one.

An earlier version of this project used a widely distributed preprocessed CSV of MIT-BIH and reported 0.9746 accuracy. That file discards record identifiers and splits at the beat level, so beats from the same patient sit in both halves and patient-level evaluation is not merely absent from it but impossible. This version measures what that costs.

---

## The result

Same network, same preprocessing, same augmentation. Only the assignment of beats to halves differs. Macro F1 over the three classes with support, with 95% cluster bootstrap intervals:

| Protocol | | macro F1 | 95% CI |
|---|---|---:|---|
| **intra** | beat-level split | **0.963** | [0.914, 0.981] |
| **inter** | de Chazal DS1/DS2 | **0.592** | [0.531, 0.626] |

The intervals are separated by 0.28. Per class the loss is not uniform: normal beats lose 0.03 and ventricular 0.17, while supraventricular and fusion beats fall from 0.91 and 0.81 to **zero**.

---

## What the project found

**Classes are nested inside patients, and that one fact explains the rest.** By inverse Simpson index the effective number of contributing recordings is 21.0 for normal beats and **1.24 for fusion** — one recording supplies 90% of the training half's fusion beats, another 75% of the test half's supraventricular ones. Across four classes the penalty orders inversely with that count.

**The recording is the sampling unit, and beat-level intervals are far too narrow.** Correctness is clustered at an intraclass correlation of 0.35, giving a design effect of 795 and an effective sample size of **62 from 49,660 beats**. On accuracy — the statistic the design-effect formula is derived for — it predicts the interval-width ratio to within 16%.

**A beat-level split hides the structure that inflates it.** The same intraclass correlation reads **0.013** under that protocol. Measured there, the data look very nearly independent and a beat-level bootstrap looks justified.

**Aggregate calibration hides where the model fails.** Expected calibration error is 0.024 overall and **0.725 on the recording the model handles worst**, where it claims 0.949 confidence at 0.224 accuracy. Low accuracy invites review; misplaced confidence does not.

**Most of what a beat-level split measures is retrievable, not learned.** Storing the training set and returning the nearest neighbour reaches 0.918 against the network's 0.963.

**Consequently nothing could be shown to help.** Five rebalancing strengths, an oversampler, five representation arms, three augmentation modes and three learning rates were compared; none improved on the simplest configuration beyond the record-level interval. Two hypotheses stated in advance were tested and refuted, the second only after two additional seeds.

---

## What it does not claim

It does not claim beat-level evaluation is wrong. It measures two protocols, not two model qualities — the intra-patient model trains on twice as many recordings and might well be the better model, which MIT-BIH has no unseen patients left to decide. A beat-level figure answers a question about adaptation while being reported as though it answered one about generalisation, and patient-adaptive classification is an established and reasonable design.

It improves nothing. Supraventricular F1 reaches 0.26 over three seeds against a published inter-patient range of 0.61 to 0.74, for reasons set out in the report: a single lead, no explicit morphological features, no band-pass filter, interval features at one scale only.

---

## Reproducing

```bash
pip install -e .

# download the raw records and build a cache (~5 min, once per arm)
python -m ecg_classification.mitdb --build-cache

# the headline comparison
python -m ecg_classification.train --protocol intra --seed 42
python -m ecg_classification.train --protocol inter --seed 42

# uncertainty, per-record decomposition, calibration
python -m analysis.bootstrap   outputs/inter-none-lossweight0-seed42
python -m analysis.records     outputs/inter-none-lossweight0-seed42 --intra outputs/intra-none-lossweight0-seed42
python -m analysis.calibration outputs/inter-none-lossweight0-seed42

# every figure in the report
python -m analysis.figures
```

Representation arms are selected by environment variable; the cache directory is derived from the arm, so an arm cannot be pointed at the wrong cache.

```bash
ECG_REPRESENTATION=wide187 python -m ecg_classification.mitdb --build-cache
ECG_REPRESENTATION=wide187 python -m ecg_classification.train --protocol inter
```

On Windows PowerShell the variable is set on its own line:

```powershell
$env:ECG_REPRESENTATION = "wide187"
python -m ecg_classification.mitdb --build-cache
python -m ecg_classification.train --protocol inter
```

| Arm | Window | Model input | Effective rate | Interval features |
|---|---|---:|---:|:---:|
| `narrow` | 0.75 s | 187 | 250 Hz | no |
| `wide187` | 1.60 s | 187 | 117 Hz | no |
| `wide400` | 1.60 s | 400 | 250 Hz | no |
| `rr_ratio` | 0.75 s | 187 | 250 Hz | yes |
| `wide187_rr` | 1.60 s | 187 | 117 Hz | yes |

---

## Layout

```
ecg_classification/
  constants.py     record partitions, AAMI mapping, representation arms
  mitdb.py         raw WFDB records to cache; protocol splits
  data.py          cache to Datasets; class weights
  augment.py       time, amplitude and noise augmentation
  model.py         1D residual CNN
  train.py         training loop, artefacts, prediction table
  predictions.py   per-beat predictions with their recording
  metrics.py       per-run scoring and artefacts
analysis/
  bootstrap.py     cluster bootstrap, ICC, design effect
  records.py       per-record decomposition, effective patient count
  calibration.py   reliability, per-recording calibration, artefact beats
  baselines.py     1-NN and logistic capacity baselines
  figures.py       every figure in the report
  eda_arms.py      what each representation arm cuts out
  eda_prematurity.py  beats aligned on the previous R peak
notebooks/
  walkthrough.ipynb   reproduces the headline numbers from stored artefacts
docs/
  report.md        the write-up
  assets/          figures, by section
```

Every run writes `predictions.csv`: one row per beat, with the recording it came from. Every analysis reads that file. Nothing downstream requires re-running inference.

---

## Data

44 of the 48 MIT-BIH recordings, the four with paced beats excluded per AAMI EC57. Beat counts match the distribution tabulated in a 2025 systematic review exactly for supraventricular and unclassifiable beats and by 39 in 100,733 overall, the difference being beats whose window would overrun a record edge.

The unclassifiable class holds 15 beats after paced-record exclusion, all of them baseline wander or electrode transients, and is held out of the model. Including it as a fifth output changes the three-class macro F1 from 0.5924 to 0.5920 and scores 0.000 itself.

---

## History

The CSV-based pipeline and its results are preserved at the tag `v0.1.0-csv` and described in Appendix A of the report. They are excluded from every comparison here because they differ in four respects at once — data source, split protocol, class definition and beat extraction — and cannot be attributed to any one.

## Reading

- [`docs/report.md`](docs/report.md) — methods, results, discussion
- `git checkout v0.1.0-csv` — the superseded pipeline

## References

The protocol is that of de Chazal, O'Dwyer and Reilly (IEEE TBME 2004); the architecture follows Kachuee, Fazeli and Sarrafzadeh (ICHI-W 2018); the resampling and design-effect procedures follow Field and Welsh (JRSS-B 2007) and Kish (1965). Full list in the report.