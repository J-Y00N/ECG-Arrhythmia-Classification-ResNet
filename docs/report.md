# Split Protocol and Representation in MIT-BIH Heartbeat Classification

> **Draft, v0.2.0-dev.** The experimental matrix is complete, including the
> representation arms. The post-hoc uncertainty analysis is not; §6 lists what
> remains. Figures have not been regenerated and are omitted.

## Abstract

An earlier version of this project reproduced a 1D residual CNN on MIT-BIH
heartbeat data and reported 0.9746 accuracy and 0.8763 macro F1. Those numbers
came from a widely used preprocessed CSV release which discards record
identifiers, and whose train and test halves stand in an exact 80:20 ratio,
indicating a beat-level rather than a patient-level split. Patient-level
evaluation is not merely absent from that release; it is impossible, because the
information required to perform it has been removed from the file.

This work rebuilds the pipeline from the raw PhysioNet records so that the
recording becomes an explicit factor in the design, and measures the consequence
of the split protocol while holding architecture and preprocessing fixed. Under
the de Chazal inter-patient partition the same network reaches a macro F1 of
0.450 ± 0.023 against 0.934 ± 0.009 under a beat-level split of the same beats:
a gap of 0.48.

The gap is not uniform. Normal beats lose 0.03 and ventricular ectopic beats
0.17, while supraventricular ectopic and fusion beats fall from 0.91 and 0.81 to
exactly zero. Minority classes are nested inside a handful of recordings -- the
effective number of contributing records, by inverse Simpson index, is 21.0 for
normal beats and 1.24 for fusion, with one recording supplying 90% of the
training half's fusion beats. And the representation excludes inter-beat timing,
which is what defines supraventricular ectopy: the previous R peak falls inside
the analysis window for 0.001% of beats.

Four representation arms were then run as a factorial design, crossing window
context against explicit normalised interval features. Widening the window
recovers ventricular beats (0.81 to 0.89) and moves supraventricular beats off
zero (0.00 to 0.09); interval features move them further (0.22) at the cost of
ventricular ones (0.71). Combining both yields no more than the window alone,
an interaction of −0.026. Across every arm the sum of the two ectopic F1 scores
stays between 0.81 and 0.99: each configuration buys one class at the other's
expense. Prematurity is a property the two ectopic classes share, at normalised
ratios of 0.763 and 0.733, so a network handed timing learns that a beat is
ectopic without learning which kind, and assigns it to whichever is four times
more common in training.

Six standard procedures are shown to fail under this structure: inverse-frequency
class weighting collapses training outright, model selection cannot be performed
on the fusion class, post-hoc prior correction cannot recover what oversampling
achieves, oversampling itself does not improve on an unweighted loss but
multiplies seed variance sevenfold, the choice of window silently deletes a
class, and combining two corrections that each work alone overshoots. A single
principle organises them: quantities measured on an absolute scale are
patient-dependent, and only ratios are portable across patients.

**Keywords:** ECG classification, MIT-BIH, inter-patient evaluation, class-patient
nesting, evaluation design, AAMI EC57

---

## 1. Introduction

### 1.1 What prompted the rebuild

The predecessor of this report used `mitbih_train.csv` and `mitbih_test.csv`
from a public preprocessed release of the MIT-BIH Arrhythmia Database. Each row
holds 187 signal samples and one label, and nothing else. Two properties of that
file motivated the present work:

1. **No record identifier.** Which recording a beat came from is not stored, so
   a patient-disjoint split cannot be constructed after the fact.
2. **An exact 80:20 train-test ratio.** Recordings contribute unequal numbers of
   beats, so a patient-wise partition cannot land on exactly 80%. The ratio is
   evidence of a beat-level split.

Together these imply that beats from the same patient appear in both halves. A
classifier can then reach the reported accuracy by recognising individual
patients' waveform idiosyncrasies rather than arrhythmia morphology, and the
reported figure carries no information about behaviour on a new patient.

### 1.2 What this work does, and does not, claim

This project does **not** replace an intra-patient evaluation with an
inter-patient one and declare the former invalid. It makes the choice between
them an experimental factor and measures its consequence, holding preprocessing,
architecture and augmentation fixed across both arms.

The earlier CSV-based result is excluded from that comparison, but not because
beat-level evaluation is meaningless. It is excluded because it differs from the
present work in four respects at once — data source, beat extraction, paced-record
handling and split protocol — and its position cannot be attributed to any one of
them. It is recorded in Appendix A as motivation, not as a comparison arm.

### 1.3 Contributions

- A pipeline rebuilt from raw WFDB records that preserves the recording identity
  of every beat, reproducing published DS1/DS2 class counts to within two beats.
- A controlled measurement of the intra- versus inter-patient gap, decomposed by
  class, with the decomposition explained by two independent mechanisms.
- A factorial design over representation, separating window context from explicit
  normalised timing, and showing the two to be redundant rather than
  complementary.
- A quantification of class-patient nesting, and a demonstration that it
  invalidates six standard procedures.
- Two hypotheses stated in advance, tested, and refuted.

---

## 2. Data

### 2.1 Source and record selection

Records are read directly from the MIT-BIH Arrhythmia Database via WFDB. The
modified limb lead II channel is selected **by name** rather than by index, which
handles record 114 — whose two signals are reversed relative to every other
recording — without a special case.

Following the AAMI EC57 recommendation, the four recordings containing paced
beats (102, 104, 107, 217) are excluded, leaving 44. Records 102 and 104 have no
MLII channel at all, surgical dressings having forced the use of V5, so they
could not have been loaded in any case.

### 2.2 Class definitions

AAMI EC57 groups the database's beat annotations into five classes (N, S, V, F,
Q). Once the paced recordings are removed, the symbols `/` and `f` disappear
entirely and the Q class retains only the literal `Q` annotation: **15 beats
across the whole database**, all of them baseline wander or electrode transients.

The model therefore has a **four-unit head**. de Chazal et al. dropped Q for this
reason and the inter-patient literature has followed since; a recent study
likewise treats a three-class (N/S/V) macro F1 as its interpretive endpoint
because only those classes carry substantial support.

This also constitutes a further reason the earlier CSV results are not
comparable. That release retained the paced recordings, so its Q class held
roughly eight thousand paced beats — a visually distinctive and easily learned
category making up about 7% of the data — rather than a handful of artefacts.

The 15 Q beats are retained as an **observation set**. They are never trained on,
never validated on and never scored, but are passed through the trained model so
that the behaviour of a classifier meeting pure artefact can be inspected. This
is a calibration question rather than a classification one; see §6.

### 2.3 Preprocessing

Only one property is inherited from the previous pipeline: a beat is 187 samples
long, because the network's classifier was built for that length. Note that this
is a choice made for comparability and not a technical constraint — global
average pooling makes the architecture independent of input length. Every other
decision below is made here on physiological grounds.

| Step | Choice | Rationale |
|---|---|---|
| Resampling | 360 → 250 Hz, `resample_poly(25, 36)` | 187 samples then span 0.748 s, roughly one cardiac cycle. Polyphase resampling applies an anti-aliasing filter |
| Window | R peak at index 62; 62 before, 125 after | 0.248 s covers the P wave and PR interval; 0.500 s covers the QRS complex and T wave |
| R-peak source | Expert annotations | Removes detector error as a confounder |
| Normalisation | Per-beat median subtraction, then division by the record's interquartile range | The first absorbs baseline wander without a filter; the second removes between-record gain differences while preserving amplitude contrast within a record |
| Boundary beats | Dropped, not zero-padded | Padding would hand the network a flat segment that means nothing |

The window deliberately does not depend on the local RR interval, so the
representation carries beat morphology and not rhythm. §4.6 examines what that
decision costs.

### 2.4 Validation against published counts

The rebuilt pipeline yields 100,694 beats across 44 records. Comparison with the
published DS1/DS2 distribution:

| Class | Published DS1 | This work, DS1 | Published total | This work, total |
|---|---:|---:|---:|---:|
| N | 45,868 | 45,848 | 90,126 | 90,088 |
| S | 942 | 944 | 2,779 | 2,781 |
| V | 3,787 | 3,788 | **7,008** | **7,008** |
| F | 415 | 414 | 803 | 802 |
| Q | 8 | 8 | **15** | **15** |

Ventricular and unclassifiable counts match exactly; the others differ by at most
two beats, accounted for by beats whose window would overrun a record boundary
and which are therefore dropped. The AAMI mapping, the record exclusions and the
DS1/DS2 partition are thus independently verified.

### 2.5 Class-patient nesting

Class frequency understates the problem. Minority-class beats are not spread
across recordings; they are concentrated in a few.

Let $p_r$ be the share of class $c$'s beats contributed by record $r$. The
**effective number of contributing records** is the inverse Simpson index

$$
N_{\text{eff}}(c) \;=\; \frac{1}{\sum_r p_r^{2}}
$$

which equals the record count under a uniform distribution and falls toward one
as a single record dominates.

| Class | DS1 beats | **N_eff** | Reading |
|---|---:|---:|---|
| N | 45,848 | **21.0** | essentially all 22 recordings |
| V | 3,788 | **7.2** | seven recordings' worth |
| S | 944 | **4.5** | four to five |
| **F** | **414** | **1.24** | **effectively one patient** |

Record 208 supplies 90% of DS1's fusion beats; record 213 supplies 93% of DS2's.
Under the inter-patient protocol, the fusion task is therefore *learn from one
patient, generalise to one other patient*. Any performance figure for that class
rests on a patient-level sample size of one.

The same statement holds for the training data as for the evaluation data: a
class does not hold as much information as its beat count suggests, because
beats within a recording are not independent observations.

---

## 3. Method

### 3.1 Protocols

Both arms consume identical preprocessing, so any difference is attributable to
the split alone.

| Protocol | Construction |
|---|---|
| **inter** | de Chazal partition: DS1 (22 records) for training, DS2 (22 records) for test. The validation half is carved out of DS1 **by record** |
| **intra** | All beats pooled, class-stratified 80:20 beat-level split, reproducing the public release's split style inside this pipeline |

Validation is split at the record level under `inter` because holding out beats
from recordings the model also trains on would reintroduce, at the point of model
selection, exactly the leakage the protocol exists to remove.

### 3.2 Representation arms

How much signal the window covers, and whether interval features accompany it,
is a factor of the experiment rather than a fixed choice. The arm is selected by
an environment variable, the cache directory is derived from it, and the run
directory is named after it, so an arm cannot be pointed at the wrong cache and
two arms cannot overwrite each other's results.

| Arm | Window | Model input | Effective rate | Intervals |
|---|---|---:|---:|:---:|
| `narrow` | 0.248 s before, 0.500 s after | 187 | 250 Hz | no |
| `wide187` | 0.80 s either side | 187 | 117 Hz | no |
| `wide400` | 0.80 s either side | 400 | 250 Hz | no |
| `rr_ratio` | as `narrow` | 187 | 250 Hz | **yes** |
| `wide187_rr` | as `wide187` | 187 | 117 Hz | **yes** |

The model input is held at 187 across the arms so that the network sees an
identically shaped input regardless of how much signal the window covers; where
the two differ the window is resampled onto it with a polyphase filter, trading
temporal resolution for context. At 0.80 s of reach the previous R peak becomes
visible for 98.6% of supraventricular beats against 55.5% of normal ones, the
widest separation available, while beats two cycles back stay under 2%.

Interval features join at the classifier rather than at the stem, so the
convolutional trunk is identical in every arm and only the classifier's input
width changes. Four features are supplied, all of them dimensionless:

| Feature | Definition |
|---|---|
| `pre_rr_ratio` | preceding interval over the local mean -- prematurity |
| `post_rr_ratio` | following interval over the local mean -- the compensatory pause |
| `local_rr_ratio` | local mean over the record median -- rate drift |
| `pre_post_ratio` | preceding over following interval |

Absolute intervals are deliberately excluded; §4.6 shows why.

### 3.3 Model

The architecture is unchanged from the version used with the CSV pipeline: a 1D
residual CNN with a convolutional stem, residual blocks, global average pooling
and a two-layer classifier head. No layer, kernel size or channel count was
modified. Every difference in results is therefore attributable to the data and
the protocol.

### 3.4 Augmentation

The reference paper does not describe an augmentation procedure, so the
augmentation here is of our own design and is not a reproduction of prior work.
Three modes are compared:

| Mode | Beat count | Operation |
|---|---|---|
| `none` | unchanged | — |
| `on_the_fly` | **unchanged** | eligible beats are *replaced* by a variant with probability 0.6 |
| `materialized` | **increased** | augmented copies of eligible beats are *appended* |

The distinction matters for attribution. `none` versus `on_the_fly` isolates
invariance injection at fixed counts; `materialized` additionally changes the
class prior, so it confounds invariance with rebalancing.

**A correction carried out during this work.** The previous augmentation clipped
its output to [0, 1], which was correct while beats were min-max normalised
inside a ten-second window. Under the present normalisation beats span roughly
−14 to +14, and that same clip saturated 9.5% of samples, flattening every R peak
and S trough — removing the QRS complex from precisely the minority-class copies
the augmentation existed to produce. Separately, time stretching moved the R peak
by up to 19 samples with a systematic bias of −3.4 samples on ventricular beats,
undoing the alignment the segmentation was designed to guarantee. Both are fixed;
the stretch now re-cuts the window around the peak's new position and pads with
the edge value, and the noise scale was re-expressed in units of the record IQR.

The general lesson is recorded because it recurs: **an augmentation procedure
cannot be specified independently of the representation it acts on.** Where the
provenance of a representation is undocumented, the augmentation built on it is
not reproducible either.

### 3.5 Rebalancing

Two mechanisms are compared.

**Loss weighting.** Weights are inverse frequency raised to an exponent $\beta$:

$$
w_c \;\propto\; \left( \frac{N}{n_c} \right)^{\beta}
$$

Weighted risk minimisation is equivalent to *unweighted* minimisation under a
tilted class prior. Writing $\pi_c$ for the class frequencies,

$$
R_w(f) \;=\; \mathbb{E}\!\left[ w_Y \, \ell\big(f(X), Y\big) \right]
\;=\; \sum_c \pi_c \, w_c \, \mathbb{E}\!\left[ \ell \mid Y = c \right]
\;\propto\; \sum_c \tilde{\pi}_c \, \mathbb{E}\!\left[ \ell \mid Y = c \right],
\qquad \tilde{\pi}_c \propto w_c \, \pi_c .
$$

Substituting $w_c \propto \pi_c^{-\beta}$ gives

$$
\tilde{\pi}_c \;\propto\; \pi_c^{\,1-\beta}.
$$

At $\beta = 0$ the weights are uniform and the model sees the true base rate; at
$\beta = 1$ the base rate is removed entirely and every class contributes an
equal share of the loss.

**Oversampling.** A weighted sampler draws indices with replacement at rates
inversely proportional to class frequency. This is *not* equivalent to loss
weighting: batches acquire a different class composition, which changes the
statistics the batch-normalisation layers learn; and Adam partially absorbs a
large loss coefficient through its per-parameter normalisation while it does not
absorb repeated gradient steps.

### 3.6 Model selection

Selection is fixed to N, S and V. Fusion cannot support it: a validation split
either takes record 208 — leaving training with about forty fusion beats — or it
does not, leaving validation with almost none. The two requirements are
incompatible and no validation size resolves it.

Deriving the scored set per split instead makes it vary by seed. At a validation
size of 0.10 one seed scored on N and S while another scored on N and V, so the
seeds were selecting their models against different objectives and averaging
across them meant nothing. Selection is restricted; **reporting is not** — test
metrics cover all four classes.

### 3.7 Experimental matrix

The `narrow` arm carries the full matrix: 3 augmentation modes × 2 protocols ×
3 seeds, with β = 0, a 100-epoch budget and early stopping at patience 15. The
`wide187` arm adds 2 protocols × 3 seeds, and the oversampler 3 further seeds
under `inter`. The remaining arms and the learning-rate sweep are single-seed
probes, and are reported as such.

Every run records whether it stopped by early stopping or by the epoch limit.
None stopped by the limit, so the budget never bound.

---

## 4. Results

### 4.1 The protocol gap

Test macro F1 at the best validation checkpoint, mean ± sd over three seeds:

| Arm | Augmentation | intra | inter | **Gap** |
|---|---|---:|---:|---:|
| `narrow` | `none` | 0.934 ± 0.009 | 0.450 ± 0.023 | **0.485** |
| `narrow` | `on_the_fly` | 0.937 ± 0.012 | 0.450 ± 0.026 | **0.487** |
| `narrow` | `materialized` | 0.940 ± 0.012 | 0.475 ± 0.037 | **0.465** |
| `wide187` | `none` | 0.953 ± 0.003 | 0.491 ± 0.003 | **0.462** |

Identical model, identical preprocessing, identical augmentation. Changing only
how beats are assigned to the two halves costs roughly half the reported macro F1.

Seed-to-seed variance is about 2.5 times larger under `inter` in the `narrow`
arm. Which recordings land in validation determines how many minority-class
beats remain for training, and because those beats sit in a few recordings the
effect is amplified.

The wider window nearly removes that instability: the inter-patient standard
deviation falls from 0.023 to 0.003, with the three seeds landing at 0.488,
0.492 and 0.493. Under `narrow` there was little generalisable signal, so which
recordings happened to be held out dominated the outcome; under `wide187` every
seed finds the same thing. The stability is the stronger evidence of the two --
more so than the 0.04 rise in the mean.

### 4.2 Per-class decomposition

Seed 42, `none`:

| Class | intra F1 | inter F1 | Δ | Morphologically distinct? | N_eff |
|---|---:|---:|---:|---|---:|
| N | 0.996 | 0.964 | −0.03 | — | 21.0 |
| V | 0.983 | 0.813 | −0.17 | **yes** — wide QRS | 7.2 |
| **S** | 0.911 | **0.000** | **−0.91** | no — QRS is normal | 4.5 |
| **F** | 0.807 | **0.000** | **−0.81** | no — intermediate | **1.24** |

Two mechanisms account for the ordering, and they are different mechanisms:

- **V survives on morphology.** A ventricular beat has a wide, distorted QRS
  complex that does not depend on knowing the patient.
- **S fails for want of timing.** A supraventricular ectopic beat arises above
  the ventricles, so ventricular conduction and hence QRS morphology are normal.
  It is defined by prematurity. With timing excluded from the representation,
  patient-specific morphology is the only remaining discriminative signal — and a
  beat-level split makes that signal available while a patient-level split does
  not. The 0.911 → 0.000 collapse is thus a direct measurement of how much of the
  intra-patient figure was patient memorisation.
- **F fails on both counts.** Fusion beats are by definition intermediate between
  normal and ventricular, so no characteristic form exists; and with N_eff = 1.24
  whatever form there is must be learned from one patient.

The fusion result is consistent with the literature rather than anomalous. A 2026
study reports F1 of 0.0659 for fusion on MIT-BIH for its proposed method and
0.0253 for an SVM baseline, and 0.0000 for that baseline on INCART, describing
the weakness as expected given the extreme rarity and heterogeneity of fusion
morphologies. Several inter-patient studies omit the class entirely.

### 4.3 Training dynamics

| Protocol | Mean best epoch |
|---|---:|
| intra | **36.7** |
| inter | **8.9** |

The two protocols do not merely differ in score; they differ in the direction of
the learning curve. Under `intra` validation performance continues improving past
epoch 40. Under `inter` it peaks between epochs 1 and 7 and then declines.

This is the memorisation mechanism visible in the optimisation trace. Continuing
to fit the training recordings is *rewarded* when those recordings also furnish
the test beats, and *paid for* when they do not.

### 4.4 Representation

Every arm was run under both protocols; the table below is seed 42 under
`inter`, where the arms differ. The three-class average over N, S and V is the
interpretive endpoint, for the reason given in §3.6.

| Arm | N | **S** | **V** | F | macro (4) | **macro (N/S/V)** |
|---|---:|---:|---:|---:|---:|---:|
| `narrow` | 0.964 | **0.000** | 0.813 | 0.000 | 0.444 | 0.592 |
| `wide187` | 0.967 | **0.090** | **0.895** | 0.000 | 0.488 | **0.650** |
| `rr_ratio` | 0.963 | **0.217** | 0.708 | 0.000 | 0.472 | 0.629 |
| `wide187_rr` | 0.970 | 0.064 | **0.924** | 0.000 | 0.489 | **0.652** |

Reported the way this literature reports, by sensitivity and positive
predictivity on the two ectopic classes:

| Arm | S Se | S +P | V Se | V +P |
|---|---:|---:|---:|---:|
| `narrow` | 0.000 | 0.000 | 0.750 | 0.888 |
| `wide187` | 0.074 | 0.115 | 0.945 | 0.849 |
| `rr_ratio` | **0.150** | **0.396** | 0.899 | 0.584 |
| `wide187_rr` | 0.040 | 0.156 | **0.964** | **0.887** |

Ventricular performance is close to published inter-patient work throughout. The
shortfall is supraventricular, and it is the subject of the rest of this section.

### 4.5 The representation grid

Crossing window context against explicit intervals gives a 2×2, in macro F1 over
four classes:

| | no intervals | with intervals |
|---|---:|---:|
| **narrow window** | 0.444 | 0.472 |
| **wide window** | 0.488 | 0.489 |

Taking `narrow` as the baseline,

$$
\begin{aligned}
\text{intervals alone} &= +0.0277 \\
\text{wider window alone} &= +0.0435 \\
\text{both} &= +0.0450 \\
\text{interaction} &= 0.0450 - 0.0277 - 0.0435 = \mathbf{-0.0262}
\end{aligned}
$$

**The interaction is negative: the two sources of timing are redundant, not
complementary.** Adding interval features to the wider window buys 0.0015 over
the window alone.

The per-class view shows this is not a plateau but a substitution. Summing the
two ectopic F1 scores:

| Arm | S | V | **S + V** |
|---|---:|---:|---:|
| `narrow` | 0.000 | 0.813 | 0.813 |
| `wide187` | 0.090 | 0.895 | 0.985 |
| `rr_ratio` | 0.217 | 0.708 | 0.925 |
| `wide187_rr` | 0.064 | 0.924 | 0.988 |

**No configuration holds both.** The sum stays between 0.81 and 0.99 while the
split between the two classes moves freely. The confusion matrices locate the
exchange: under `rr_ratio` the count of supraventricular beats called
ventricular rises from 354 to 720, and normal beats called ventricular from 154
to 1,327.

The mechanism is in the interval features themselves (§4.6): prematurity is
almost identical for the two ectopic classes, at 0.763 and 0.733 of the local
mean. Timing tells the network that a beat is ectopic. It does not tell it which
kind, and the network resolves the ambiguity toward the class that is four times
more frequent in training.

### 4.6 Interval features and the absolute-scale illusion

Mean feature values by class, over the whole database:

| | `pre_rr_ratio` | `post_rr_ratio` | `local_rr_ratio` | `pre_post_ratio` |
|---|---:|---:|---:|---:|
| N | 1.028 | 0.983 | 1.006 | 1.068 |
| **S** | **0.763** | 1.044 | 1.120 | 0.821 |
| **V** | **0.733** | **1.202** | 1.024 | 0.653 |
| **F** | **0.983** | 0.906 | 1.020 | 1.113 |

Three things follow.

**The features recover a textbook distinction.** Ventricular ectopic beats carry
a markedly longer following interval than supraventricular ones (1.202 against
1.044). A ventricular beat does not reset the sinus node and is followed by a
full compensatory pause; a supraventricular beat does reset it and is followed
by an incomplete one. That the separation appears unprompted is evidence the
features measure what they are meant to.

**Prematurity does not separate the ectopic classes.** At 0.763 and 0.733 the two
are nearly identical, which is why §4.5 finds a substitution rather than a gain.

**Fusion beats are not premature.** On an absolute scale they appear to be -- a
median preceding interval of 0.564 s against 0.768 s for normal beats -- but
normalised they sit at 0.983, on top of the normal distribution. Records 208 and
213 supply almost all of them, and those recordings run at 103 and 108 beats per
minute against 75 and 69 for typical records. The fusion beats are not early;
the patients who have them are fast.

A model given absolute timing would therefore learn to detect fusion as *a
patient whose heart rate is high*. That is patient recognition, and it would
transfer from DS1 to DS2 here only because records 208 and 213 happen to share
the trait. The `wide187` arm, which carries absolute timing implicitly, never
predicted fusion at all -- which is the correct behaviour and confirms the
prediction made before the run.

### 4.7 Rebalancing strength

Loss weighting was swept over β (inter, seed 42):

| β | Weight ratio | Loss share (N/S/V/F) | train acc | test macro F1 |
|---:|---:|---|---:|---:|
| 0 | 1.0 | 89.5 / 1.2 / 8.3 / 0.9 % | 0.977 | **0.450** |
| 0.25 | 3.1 | 80.6 / 3.2 / 13.6 / 2.6 % | 0.980 | 0.448 |
| 0.5 | 9.9 | 65.7 / 7.7 / 20.0 / 6.6 % | 0.971 | 0.449 |
| 0.75 | 31.3 | 45.2 / 15.5 / 25.0 / 14.4 % | 0.884 | 0.391 |
| **1.0** | **98.4** | **25 / 25 / 25 / 25 %** | **0.118** | **0.220** |

At β = 1 — plain inverse frequency, the textbook default — training collapses
within a single epoch. Equalising the total loss contribution of every class
means 414 fusion beats carry the same weight as 40,753 normal beats, and since
those 414 come from essentially one patient the network cannot learn to
discriminate them; predicting the rare classes indiscriminately becomes the
faster way to reduce the weighted loss.

No degree of rebalancing improves on β = 0. **How much rebalancing a dataset
tolerates is itself a measurement of how deeply its classes are nested inside
individual recordings**, and is reported here as a result rather than tuned away.

Oversampling is treated separately in §4.8, where a single-seed result that
appeared to beat the unweighted loss did not survive two more seeds.

### 4.8 Two refuted hypotheses

Because the tilt introduced by rebalancing is a known quantity, it can be undone
at prediction time without retraining:

$$
p_{\beta}(c \mid x) \;\propto\; \frac{p_{\text{model}}(c \mid x)}{\tilde{\pi}_c^{\,\beta}}
$$

**Hypothesis, stated before the test.** The oversampler's advantage is a shift of
the decision threshold rather than a difference in what was learned; applying the
correction above to the *unweighted* model's stored probabilities should therefore
recover most of it.

**Result.** It does not.

| β | macro F1 | N | S | V | F |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.444 | 0.964 | 0.000 | 0.813 | 0.000 |
| 0.5 | 0.441 | 0.953 | 0.003 | 0.809 | 0.000 |
| 0.75 | 0.425 | 0.851 | 0.021 | 0.827 | 0.000 |
| 1.0 | 0.307 | 0.324 | 0.084 | 0.815 | 0.005 |
| 1.25 | 0.225 | 0.000 | 0.079 | 0.785 | 0.036 |

*(oversampler reference: 0.521, S F1 with recall 0.67)*

Macro F1 decreases monotonically and supraventricular F1 never exceeds 0.084.
The hypothesis is refuted, and the reason is instructive: **prior correction is a
monotone transformation of the model's scores, so it can only redistribute
decisions along an existing ranking.** In the unweighted model the
supraventricular score is almost always below the normal score, and no threshold
recovers a separation that is not present. Oversampling must therefore have
changed the ranking itself — consistent with the batch-composition and
gradient-step arguments of §3.4 rather than with a threshold effect.

### 4.9 Timing information audit

Pre-RR intervals were computed from the stored R-peak positions, which are part
of the cache and carry no class information.

| Class | median pre-RR (s) | 25th pct |
|---|---:|---:|
| N | 0.768 | 0.672 |
| **S** | **0.676** | **0.488** |
| V | 0.512 | 0.452 |
| F | 0.564 | 0.548 |

The window reaches 0.248 s back from the R peak. **The previous R peak falls
inside it for 0.0010% of beats** — one beat in a hundred thousand. Prematurity is
present in the data, is discriminative, and is excluded by construction.

Widening the window would admit it, at a cost. The fraction of beats whose
previous R peak becomes visible, by window reach:

| Reach (s) | N | S | V | F | Two beats back |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 2.1% | 27.7% | 45.6% | 1.1% | 0.0% |
| 0.65 | 19.7% | 46.7% | 90.6% | 93.3% | 0.1% |
| **0.80** | **55.5%** | **98.6%** | 94.6% | 98.1% | 1.6% |
| 1.00 | 83.8% | 99.5% | 98.2% | 98.8% | 5.9% |

The N–S separation is widest near 0.80 s.

**However, an absolute window cannot carry portable timing.** The number of
neighbouring beats visible in a fixed-duration window is itself the patient's
resting heart rate.

The point is made sharply by the fusion class. On an absolute scale fusion beats
look premature — a median pre-RR of 0.564 s against 0.768 s for normal beats. But
almost all of them come from two recordings, and those two recordings are fast:

| Record | Median RR | Rate | Role |
|---|---:|---:|---|
| **208** | 0.580 s | **103 bpm** | supplies 90% of DS1 fusion beats |
| **213** | 0.556 s | **108 bpm** | supplies 93% of DS2 fusion beats |
| 100 | 0.796 s | 75 bpm | typical |
| 103 | 0.868 s | 69 bpm | typical |

The fusion median of 0.564 s is indistinguishable from the resting interval of
the two recordings that produce it. **Fusion beats are not early; the patients who
have them are fast.** Normalised by each recording's local mean RR they sit at
1.0, on top of the normal distribution.

A model given absolute timing would therefore learn to detect fusion as *"a
patient whose heart rate is high"*. That is patient recognition, not arrhythmia
recognition, and it would transfer from DS1 to DS2 here only because records 208
and 213 happen to share the trait — an accident of the partition rather than
generalisation.

This is why de Chazal separated the two: morphology from an absolute window,
timing from separately normalised interval features. The present work arrived at
the same conclusion independently, by a different route.

### 4.10 Learning rate

The inter-patient runs reach their best validation epoch early -- a mean of 8.9
against 36.7 for intra-patient -- which invites the objection that the runs are
simply undertrained. Three learning rates were tried on `wide187`, seed 42:

| Learning rate | Best epoch | **Train accuracy at that epoch** | macro (N/S/V) |
|---:|---:|---:|---:|
| 3e-4 | 21 | **0.9951** | 0.660 |
| 1e-3 | 16 | **0.9960** | 0.650 |
| 3e-3 | 11 | **0.9944** | 0.673 |

Training accuracy stands above 99.4% at the validation peak under every setting,
so the network is not short of fitting; and the peak moves by a factor of two
across the sweep while the score moves by 0.023. The learning rate changes the
trajectory and not the destination.

The early peak is therefore not an optimisation artefact. It is the point at
which the generalisable signal in the training recordings runs out, after which
what the network continues to learn is specific to the patients it trained on.
The same reading is supported by the arms: the best epoch under `inter` rises
from 1 to 16 when the window widens, because widening it gives the network more
that transfers.

Worth noting against ourselves: 3e-3 scores highest here, so the default of 1e-3
is not demonstrably optimal. With one seed and a spread of 0.023 against a
seed-level standard deviation of 0.004 in this arm, the ordering is suggestive
rather than established.

### 4.11 Sensitivity: records 201 and 202

Records 201 and 202 come from the same male subject, and the de Chazal partition
places 201 in DS1 and 202 in DS2. The published inter-patient split is therefore
not strictly subject-disjoint. Excluding record 202 changes test macro F1 from
0.4443 to 0.4442. The caveat is real and its effect is negligible.

---

## 5. Discussion

### 5.1 Six failures of standard procedure

| # | Procedure | Failure |
|---|---|---|
| 1 | Inverse-frequency class weighting | Collapses training; no exponent improves on none |
| 2 | Validation-based model selection | Impossible for fusion — training and validation support are mutually exclusive |
| 3 | Post-hoc prior correction | Cannot recover the oversampler's gain; a ranking that does not separate cannot be thresholded into one |
| 4 | Fixed-window representation | Silently deletes a class whose definition is temporal |

None of these is a bug in the implementation. Each is a consequence of the same
data structure, and each would have gone unnoticed under a beat-level split.

### 5.2 Absolute and normalised scales

Three axes of this project independently reached the same conclusion.

| Axis | Absolute — patient dependent | Normalised — patient invariant |
|---|---|---|
| **Amplitude** | raw millivolts | divided by the recording's IQR |
| **Sample size** | beat count | `N_eff`, the inverse Simpson index |
| **Timing** | pre-RR in seconds | pre-RR divided by the local mean |

That the same distinction emerged three times, from three unrelated starting
points, suggests it is a property of the data rather than of any one analysis.

### 5.3 Class imbalance and patient heterogeneity are one structure

These are usually treated as separate problems. In this database they are not:
minority-class beats are unevenly nested inside recordings, so any operation on
the class axis is also an operation on the patient axis. Oversampling a class
replicates a patient. Rebalancing a loss reweights a patient. `N_eff` is the
quantity that makes the coupling explicit, and it is what explains why the class
ordering of the protocol gap is not the class ordering of the frequencies.

The oversampling result in §4.8 is the sharpest illustration. Replicating fusion
beats replicates record 208, so a run's outcome comes to depend on whether that
recording survived the validation split -- which is exactly the seven-fold rise
in variance that appeared once two more seeds were run. The procedure did not
add patient diversity. It amplified the dependence on the diversity already
present.

### 5.4 Limitations

- **Supraventricular performance is below published inter-patient work**, at a
  best F1 of 0.22 against a literature range of roughly 0.39 to 0.69. Three
  design choices account for it, all of them deliberate. The morphology
  representation is a single lead with no explicit features -- no QRS duration,
  no P-wave shape -- and no band-pass filtering, so the low-amplitude P wave that
  distinguishes an atrial ectopic beat is present in the window but weak.
  Interval features come from one local window, where the literature uses several
  scales including a five-minute average. And no rebalancing is applied, because
  none improved the aggregate. What this study measures is the effect of the
  split protocol, and that effect is consistent across every representation arm.
- **Expert R-peak annotations are assumed.** Substituting an automatic detector
  would introduce localisation error that this work does not model.
- **Subject, session and electrode placement are perfectly confounded.** The
  database holds one recording per subject, so the components of between-recording
  variation cannot be separated even in principle.
- **Augmentation confounds two effects in the `materialized` mode.** Its
  class-dependent multipliers change the prior as well as injecting invariance;
  the two are separable only by a factorial design not run here.
- **Three of the five representation arms are single-seed.** The grid and the
  interaction estimate rest on one seed each for `rr_ratio` and `wide187_rr`. A
  seed-level standard deviation of 0.004 in the `wide187` arm suggests the
  ordering is stable, but the oversampling result in §4.8 is a reminder of what
  a single seed can hide.
- **The learning rate is not demonstrably optimal.** 3e-3 scored above the 1e-3
  default in a single-seed sweep.
- **Fusion figures are unstable at any protocol.** A patient-level sample size of
  one does not support a class-wise estimate.
- **The 15 unclassifiable beats are observed, not scored.**

---

## 6. Pending work

The experimental matrix is complete. What remains is the post-hoc analysis,
which requires no further training: every figure below is computed from the
prediction tables already written.

**Uncertainty quantification.** Beats are clustered within recordings, so a
beat-level bootstrap treats roughly fifty thousand correlated observations as
independent and will understate the confidence interval. The resampling unit
should be the recording.

Let $Y_{ij}$ be the correctness of beat $j$ in recording $i$, modelled as

$$
Y_{ij} = \mu + a_i + \varepsilon_{ij}, \qquad
a_i \sim \left(0, \sigma_a^{2}\right), \quad
\varepsilon_{ij} \sim \left(0, \sigma_\varepsilon^{2}\right),
$$

so that the intraclass correlation and the resulting design effect are

$$
\rho \;=\; \frac{\sigma_a^{2}}{\sigma_a^{2} + \sigma_\varepsilon^{2}},
\qquad
D_{\text{eff}} \;=\; 1 + (\bar{m} - 1)\,\rho,
\qquad
n_{\text{eff}} \;=\; \frac{n}{D_{\text{eff}}},
$$

with $\bar{m}$ the mean number of beats per recording. Estimating $\rho$ by REML
or by the method of moments requires no sampling, and $D_{\text{eff}}$ predicts
the ratio between the two interval widths — converting the argument for the
cluster bootstrap from a claim into a check.

**Capacity baselines.** An L2-regularised multinomial logistic regression on the
same input and split — which is the MAP estimate under a Gaussian prior, so the
capacity axis is also a prior-strength axis — and a 1-nearest-neighbour
classifier as the memorisation-only extreme. The prediction is that the
protocol gap widens with capacity.

**Per-record decomposition and N_eff correlation.** Whether the inter-patient
penalty is spread evenly across the 22 test recordings or concentrated in a few,
and whether the class-wise penalty tracks `N_eff`.

**Calibration.** Reliability curves and expected calibration error under both
protocols, and the behaviour of the trained model on the 15 artefact beats. A
classifier that is confidently wrong on a detached-electrode waveform is a
clinically worse failure than one that is uncertain.

**Uncertainty on the arm comparison.** The representation grid is reported as
point estimates. The cluster bootstrap will say which of its differences survive,
and the single-seed arms need either more seeds or an explicit interval.

## 7. Future work

- A hierarchical model with the recording as a random effect would separate the
  variance components of class and patient effects, formalising the observation
  that oversampling does not increase effective patient diversity. Partial
  pooling would shrink the estimate for a class supported by one patient and
  widen its uncertainty automatically — which is what loss weighting attempted
  here by an indirect route, and failed at.
- A factorial design separating prior rebalancing from invariance injection.
- Physiologically structured time warping: uniform stretching scales the QRS
  complex along with everything else, whereas heart-rate variation compresses the
  diastolic interval and shortens the QT interval while leaving QRS duration
  largely unaffected.
- Extension to INCART or the MIT-BIH Supraventricular database, to test whether
  the nesting structure is a property of this database or of ambulatory
  arrhythmia data in general.
- Transfer of nuisance parameters across recordings as an augmentation. Note that
  this, like any generative approach, cannot raise `N_eff`: a model fitted to one
  patient's fusion beats produces that patient's fusion beats. **Information that
  is absent cannot be synthesised.**

---

## Appendix A. Superseded results

The previous version of this project used the public preprocessed CSV release and
reported 0.9746 accuracy and 0.8763 macro F1 over five classes, with materialized
augmentation, on a beat-level split.

Those results are **not comparable** with the present ones and are excluded from
every comparison in §4. They differ in four respects simultaneously:

1. **Data source** — a preprocessed CSV of undocumented provenance rather than
   the raw records.
2. **Split protocol** — beat-level rather than record-level.
3. **Class definition** — five classes including a Q class of roughly eight
   thousand paced beats, rather than four.
4. **Beat extraction and normalisation** — a rate-dependent window with min-max
   scaling, rather than a fixed window with robust scaling.

They are recorded here because they are the reason this work exists, not as a
baseline. The code and artefacts that produced them are preserved at the git tag
`v0.1.0-csv`.

---

## References

1. Kachuee M, Fazeli S, Sarrafzadeh M. *ECG Heartbeat Classification: A Deep
   Transferable Representation.* IEEE ICHI-W, 2018. arXiv:1805.00794.
2. de Chazal P, O'Dwyer M, Reilly RB. *Automatic Classification of Heartbeats
   Using ECG Morphology and Heartbeat Interval Features.* IEEE Trans Biomed Eng.
   2004;51(7):1196–1206.
3. Moody GB, Mark RG. *The Impact of the MIT-BIH Arrhythmia Database.* IEEE Eng
   Med Biol Mag. 2001;20(3):45–50. DOI: 10.1109/51.932724.
4. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.
   2000;101(23):e215–e220.
5. MIT-BIH Arrhythmia Database. PhysioNet. https://physionet.org/content/mitdb/1.0.0/
   DOI: 10.13026/C2F305.
6. ANSI/AAMI EC57. *Testing and Reporting Performance Results of Cardiac Rhythm
   and ST Segment Measurement Algorithms.* 1998.
7. Huang H, Liu J, Zhu Q, Wang R, Hu G. *A New Hierarchical Method for
   Inter-Patient Heartbeat Classification Using Random Projections and RR
   Intervals.* BioMedical Engineering OnLine. 2014;13:90.
8. Elkan C. *The Foundations of Cost-Sensitive Learning.* IJCAI, 2001.
9. *DeepArrhythmia: Segment-Contextualized ECG Arrhythmia Classification via
   Selective Evidence Acquisition.* arXiv:2605.16441. *(cited for fusion-class
   performance; verify against the source before final submission)*