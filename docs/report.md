# Measuring the Inter-Patient Penalty in MIT-BIH Heartbeat Classification

> **Draft, v0.2.0-dev.** The experimental matrix and the post-hoc analysis are
> complete. Figures have not been regenerated and are omitted; exploratory
> figures for the wide-window arms are still outstanding.

## Abstract

That patient-level evaluation is harder than beat-level evaluation has been
known in this field since de Chazal et al. established the DS1/DS2 partition in
2004. The question this study asks is not how much harder, but whether the tools
ordinarily used can measure it. Largely they cannot.

An earlier version of this project reproduced a 1D residual CNN on a widely used
preprocessed release of MIT-BIH and reported 0.9746 accuracy and 0.8763 macro
F1. That release discards record identifiers and splits at the beat level, so
patient-level evaluation is not merely absent from it but impossible. The
pipeline was rebuilt from the raw records to make the recording an explicit
factor, reproducing the published DS1/DS2 class counts to within two beats.

Separating patients costs about half the macro F1: 0.593 against 0.963 over the
three scoreable classes, with cluster bootstrap intervals that do not overlap.
The loss is not uniform. Normal beats lose 0.03 and ventricular ectopic beats
0.17, while supraventricular and fusion beats fall to zero. Minority classes are
nested inside a handful of recordings -- by inverse Simpson index the effective
number of contributing records is 21.0 for normal beats and 1.24 for fusion, one
recording supplying 90% of the training half's fusion beats and another 75% of
the test half's supraventricular ones. Across four classes the per-class penalty
orders inversely with that count.

Three findings concern measurement rather than performance. Beats are clustered
within recordings at an intraclass correlation of 0.35, giving a design effect
of 795 and an effective sample size of 64 from 49,660 beats; beat-level
confidence intervals are consequently 16 times too narrow, and the design-effect
prediction is confirmed to within 16% on accuracy, the statistic for which the
formula is derived. A beat-level split hides the very structure that inflates
it, the same intraclass correlation reading 0.013 under that protocol. And
aggregate calibration hides where the model fails: an expected calibration error
of 0.024 overall against 0.725 on the single recording the model handles worst,
where it claims 0.949 confidence at 0.224 accuracy.

Consequently no intervention could be shown to help. Five rebalancing
strengths, an oversampler, four representation arms and three learning rates
were compared; every difference fell inside the record-level interval. Two
hypotheses stated in advance -- that post-hoc prior correction would recover
what oversampling achieved, and that oversampling raised the ceiling -- were
tested and refuted, the second only after two additional seeds. Twenty-two
recordings are not enough to distinguish these interventions, and the interval
that shows this is the same one that shows the protocol gap to be real.

**Keywords:** ECG classification, MIT-BIH, inter-patient evaluation, cluster
bootstrap, design effect, class-patient nesting, AAMI EC57

---

## 1. Introduction

### 1.1 The task as signal and nuisance

A beat classifier has to separate a pathological signal from patient-specific
nuisance variation. Amplitude varies with thoracic impedance, cardiac axis and
body habitus; intervals vary with resting heart rate; both vary with electrode
placement and recorder gain. None of that is what the labels describe, and every
normalisation step in this pipeline is an attempt to divide the two.

What this study finds is that when classes are nested inside patients the
division becomes impossible in principle. For a class whose beats come from one
recording there is no sample against which to distinguish that patient's
pathology from that patient's idiosyncrasy.

### 1.2 What prompted the rebuild

The predecessor to this report used a public preprocessed release of the MIT-BIH
Arrhythmia Database. Each row holds 187 signal samples and one label. Two
properties of that file motivated the present work:

1. **No record identifier.** Which recording a beat came from is not stored, so
   a patient-disjoint split cannot be constructed after the fact.
2. **An exact 80:20 train-test ratio.** Recordings contribute unequal numbers of
   beats, so a patient-wise partition cannot land on exactly 80%.

Together these imply beats from the same patient in both halves. A classifier
can then reach the reported accuracy by recognising individual patients'
idiosyncrasies, and the figure carries no information about a new patient.

This is not an unusual situation. A 2025 systematic review of 122 ECG
classification studies published between 2017 and 2024 found that 96 used the
MIT-BIH database, 68 followed the AAMI recommendations, and **37 -- under a
third -- adopted the inter-patient paradigm**. The protocol has been available
since 2004; most of the field does not use it.

### 1.3 What this work does and does not claim

This project does not replace an intra-patient evaluation with an inter-patient
one and declare the former invalid. It makes the choice between them an
experimental factor and measures its consequence, holding preprocessing,
architecture and augmentation fixed across both arms.

The earlier CSV-based result is excluded from that comparison, but not because
beat-level evaluation is meaningless. It is excluded because it differs from the
present work in four respects at once -- data source, beat extraction,
paced-record handling and class definition -- and its position cannot be
attributed to any one of them. It appears in Appendix A as motivation.

It should also be said that this study improves nothing. No configuration
classified supraventricular or fusion beats reliably. That is reported as the
result it is, together with the interval that explains why a reliable
improvement could not have been recognised here even if one had been found.

### 1.4 Contributions

- A pipeline rebuilt from raw WFDB records preserving the recording identity of
  every beat, validated against published DS1/DS2 counts.
- A controlled measurement of the protocol gap, decomposed by class, with the
  decomposition explained by two independent mechanisms.
- A quantification of class-patient nesting, and the observation that the
  per-class penalty orders inversely with it.
- Intraclass correlation and design effect for this task, with the design-effect
  prediction validated on accuracy and used to justify the resampling unit.
- A capacity axis -- nearest neighbour, linear, convolutional -- showing that
  most of what a beat-level split measures is retrievable rather than learned.
- Two hypotheses stated in advance, tested, and refuted.

---

## 2. Data

### 2.1 Source and record selection

Records are read directly from the MIT-BIH Arrhythmia Database via WFDB. The
modified limb lead II channel is selected **by name** rather than by index,
which handles record 114 -- whose two signals are reversed relative to every
other recording -- without a special case.

Following AAMI EC57, the four recordings containing paced beats (102, 104, 107,
217) are excluded, leaving 44. Records 102 and 104 have no MLII channel at all,
surgical dressings having forced the use of V5.

AAMI also recommends excluding segments of ventricular flutter and fibrillation.
Flutter waves carry the annotation `!`, which belongs to no AAMI superclass and
is therefore dropped during segmentation along with the other non-beat
annotations: 472 such waves are removed from record 207. This is exclusion by
symbol rather than by segment, so a beat annotated normal or ventricular inside
a flutter episode would remain; agreement with the published DS1/DS2 totals
suggests the number is small.

### 2.2 Class definitions

AAMI EC57 groups the database's annotations into five classes. Once the paced
recordings are removed the symbols `/` and `f` disappear entirely and the Q class
retains only the literal `Q` annotation: **15 beats across the whole database**,
all of them baseline wander or electrode transients.

The model therefore has a four-unit head. This departs from de Chazal et al.,
who classified into all five AAMI classes, and follows instead a practice common
in later inter-patient work, where Q is dropped on the grounds that what remains
of it is not representative of anything a classifier is meant to detect.
Restricting the interpretive endpoint further, to the three classes carrying
substantial support, also has precedent.

This is a further reason the earlier results are not comparable. That release
retained the paced recordings, so its Q class held roughly eight thousand paced
beats -- a visually distinctive category making up about 7% of the data -- rather
than a handful of artefacts.

The 15 Q beats are retained as an **observation set**, never trained on,
validated on or scored, but passed through the trained model (§4.9).

The exclusion is verified rather than argued. Running the same configuration
with Q as a fifth output class moves the macro F1 over the three supported
classes from 0.5924 to 0.5920, and Q itself scores 0.000. Giving an output unit
to fifteen artefact beats changes nothing for the other classes and the unit
learns nothing. Since de Chazal et al. classified into five classes, this also
establishes that the two configurations agree on the classes that carry
support.

### 2.3 Preprocessing

| Step | Choice | Rationale |
|---|---|---|
| Resampling | 360 to 250 Hz, polyphase | 187 samples then span 0.748 s, roughly one cardiac cycle; the filter prevents aliasing |
| Window | R peak at index 62; 62 before, 125 after | 0.248 s covers the P wave and PR interval; 0.500 s covers the QRS complex and T wave |
| R-peak source | Expert annotations | Removes detector error as a confounder |
| Normalisation | Per-beat median subtraction, then division by the record's interquartile range | The first absorbs baseline wander without a filter; the second removes between-record gain differences while preserving amplitude contrast within a record |
| Boundary beats | Dropped, not zero-padded | Padding would hand the network a flat segment that means nothing |

One point should be made explicitly, because the limitations in §5.7 are easy to
misread without it. A single-lead recording is one potential against time. No
step of this pipeline discards spatial or inter-lead phase information: such
information does not exist in the source and would require a multi-lead
recording to obtain. What §5.7 discusses is present in the raw signal and absent
from this representation, which is a different thing from having been destroyed
by it.

### 2.4 Validation against published counts

100,694 beats across 44 records, against the distribution tabulated in a 2025
systematic review of this literature:

| Class | Reference DS1 | Here, DS1 | Reference total | Here, total |
|---|---:|---:|---:|---:|
| N | 45,866 | 45,848 | 90,125 | 90,088 |
| **S** | **944** | **944** | **2,781** | **2,781** |
| V | 3,788 | 3,788 | 7,009 | 7,008 |
| F | 415 | 414 | 803 | 802 |
| **Q** | **8** | **8** | **15** | **15** |
| total | 51,021 | 51,002 | 100,733 | 100,694 |

Supraventricular and unclassifiable counts match exactly in both halves;
ventricular and fusion differ by one beat each, and normal by 37, for a total
shortfall of 39 in 100,733. These are beats whose window would overrun a record
boundary and which are dropped rather than zero-padded. The AAMI mapping, the
record exclusions and the partition are thus independently verified.

### 2.5 Class-patient nesting

Class frequency understates the problem. Let $p_r$ be the share of class $c$'s
beats contributed by record $r$. The **effective number of contributing
records** is the inverse Simpson index

$$
N_{\text{eff}}(c) = \frac{1}{\sum_r p_r^{2}}
$$

which equals the record count under a uniform distribution and falls toward one
as a single record dominates.

| Class | DS1 beats | $N_{\text{eff}}$ | dominant | DS2 $N_{\text{eff}}$ | dominant |
|---|---:|---:|---|---:|---|
| N | 45,848 | **21.01** | 215 (7%) | 20.71 | 212 (6%) |
| V | 3,788 | **7.15** | 208 (26%) | 5.71 | 233 (26%) |
| S | 944 | **4.48** | 209 (41%) | **1.72** | **232 (75%)** |
| F | 414 | **1.24** | **208 (90%)** | **1.15** | **213 (93%)** |

Under the inter-patient protocol the fusion task is *learn from one patient,
generalise to one other patient*, and the supraventricular task is not much
better on the evaluation side. Any figure for those classes rests on a
patient-level sample size near one.

---

## 3. Method

### 3.1 Protocols

Both arms consume identical preprocessing, so any difference is attributable to
the split alone.

| Protocol | Construction |
|---|---|
| **inter** | de Chazal partition: DS1 (22 records) for training, DS2 (22) for test. The validation half is carved out of DS1 **by record** |
| **intra** | All beats pooled, class-stratified 80:20 beat-level split |

Validation is split at the record level under `inter` because holding out beats
from recordings the model also trains on would reintroduce, at the point of
model selection, the leakage the protocol exists to remove.

de Chazal et al. also separated selection from assessment, using DS1 to choose
among twelve candidate classifier configurations and DS2 for an independent
assessment of the chosen one. The difference here is where the boundary falls:
their selection consumed all of DS1, whereas a network trained by gradient
descent needs a held-out set during training, so DS1 is divided again. In both
arrangements DS2 is untouched until the end.

### 3.2 Representation arms

How much signal the window covers, and whether interval features accompany it,
is a factor of the experiment. The arm is selected by an environment variable
and the cache directory is derived from it, so an arm cannot be pointed at the
wrong cache.

| Arm | Window | Model input | Effective rate | Intervals |
|---|---|---:|---:|:---:|
| `narrow` | 0.248 s before, 0.500 s after | 187 | 250 Hz | no |
| `wide187` | 0.80 s either side | 187 | **117 Hz** | no |
| `wide400` | 0.80 s either side | **400** | 250 Hz | no |
| `rr_ratio` | as `narrow` | 187 | 250 Hz | **yes** |
| `wide187_rr` | as `wide187` | 187 | 117 Hz | **yes** |

At 0.80 s of reach the previous R peak becomes visible for 98.6% of
supraventricular beats against 55.5% of normal ones, the widest separation
available, while beats two cycles back stay under 2%. Where the window exceeds
the input length it is resampled onto it with a polyphase filter, trading
temporal resolution for context; `wide400` holds the same window at full rate
and exists to say which of the two mattered.

Interval features join at the classifier rather than at the stem, so the
convolutional trunk is identical in every arm and only the classifier's input
width changes. All four are dimensionless: preceding and following interval over
the local mean, local mean over the record median, and the ratio of the first
two. Absolute intervals are deliberately excluded; §4.8 shows why.

### 3.3 Model

The classifier is a 1D convolutional network with residual connections. A stem
maps the single input channel to 32 channels with a kernel-5 convolution,
batch normalisation and ReLU. Five residual blocks follow; each holds two
kernel-5 convolutions with batch normalisation, adds the block input to the
second convolution's output, and closes with ReLU, a kernel-5 stride-2 max pool
and dropout at 0.10. Global average pooling then reduces the sequence to one
value per channel, and a two-layer head maps 32 to 64 to the class logits with
ReLU and dropout at 0.20 between them.

The design follows the residual heartbeat architecture of Kachuee et al.,
retaining the modifications to normalisation, regularisation and classifier
structure made in the earlier version of this project. The pooling deserves a
note because it bears on §4.6: global average pooling discards where along the
sequence a feature occurred, so the network reads the separation between two
features but not the absolute position of one. A representation that encoded
prematurity as a shift would therefore be invisible to it, while one that
encodes it as the gap between two R peaks is not.

**No layer, kernel size or channel count was changed in this study.** The
convolutional trunk is byte-identical across every representation arm, and in
the arms that supply interval features only the head's first layer widens, from
32 inputs to 36. Where no interval features are supplied the forward pass and
the parameter names match the version used with the CSV pipeline exactly, so
checkpoints from those runs load without translation. Every difference in
results is therefore attributable to the data and the protocol.

### 3.4 Augmentation

The reference paper does not describe an augmentation procedure, so what is here
is our own design. Three modes were compared under `narrow`: `none`,
`on_the_fly` (eligible beats replaced by a variant, count unchanged) and
`materialized` (augmented copies appended, count increased). The distinction
matters for attribution -- the first contrast isolates invariance injection at
fixed counts, the second additionally changes the class prior.

A correction made during this work is worth recording. The previous augmentation
clipped its output to [0, 1], correct while beats were min-max normalised inside
a ten-second window. Under the present normalisation beats span roughly -14 to
+14, and that clip saturated 9.5% of samples, flattening every R peak and S
trough -- removing the QRS complex from precisely the minority-class copies the
augmentation existed to produce. Time stretching separately moved the R peak by
up to 19 samples with a systematic bias of -3.4 on ventricular beats. Both are
fixed. The general lesson recurs: an augmentation procedure cannot be specified
independently of the representation it acts on.

Augmentation is disabled in every arm beyond `narrow`.

### 3.5 Rebalancing

Weights are inverse frequency raised to an exponent $\beta$. Weighted risk
minimisation is equivalent to unweighted minimisation under a tilted class
prior: writing $\pi_c$ for the class frequencies,

$$
R_w(f) = \sum_c \pi_c w_c \, \mathbb{E}[\ell \mid Y = c]
\propto \sum_c \tilde{\pi}_c \, \mathbb{E}[\ell \mid Y = c],
\qquad \tilde{\pi}_c \propto w_c \pi_c
$$

and with $w_c \propto \pi_c^{-\beta}$ this gives $\tilde{\pi}_c \propto
\pi_c^{1-\beta}$. At $\beta = 1$ the base rate is removed entirely.

An oversampler is compared separately. It is not equivalent: batches acquire a
different class composition, which changes what the batch-normalisation layers
learn, and Adam partially absorbs a large loss coefficient through its
per-parameter normalisation while it does not absorb repeated gradient steps.

### 3.6 Model selection

Selection is fixed to N, S and V. Fusion cannot support it: a validation split
either takes record 208 -- leaving training with about forty fusion beats -- or
it does not, leaving validation with almost none. Deriving the scored set per
split instead made it vary by seed, so seeds were selecting against different
objectives and averaging across them meant nothing. Selection is restricted;
reporting is not.

### 3.7 Capacity baselines

Two models bracket the network on a capacity axis, on identical inputs and
splits. **1-nearest neighbour** stores the training set and answers with its
closest member: it memorises and does nothing else. **L2-regularised multinomial
logistic regression** fits one linear boundary per class and has almost no
capacity to memorise; being the MAP estimate under a Gaussian prior, its
regularisation strength is also a prior precision. Neither is a competitor to
the network; both are instruments for measuring what its advantage is made of.

### 3.8 Experimental matrix

`narrow` carries the full matrix: 3 augmentation modes x 2 protocols x 3 seeds,
$\beta = 0$, a 100-epoch budget and early stopping at patience 15. `wide187` and
`wide400` add 2 protocols x 3 seeds, the oversampler 3 seeds under `inter`, and
`rr_ratio` and `wide187_rr` one seed each. A learning-rate sweep and the two
baselines are single-seed probes and are reported as such. No run stopped at the
epoch limit.

### 3.9 Uncertainty

Beats are clustered within recordings, so a bootstrap that resamples beats
treats correlated observations as independent. Uncertainty is therefore
estimated by a **cluster bootstrap** [Field and Welsh 2007]: the resampling unit
is the recording, and a recording drawn twice contributes all of its beats
twice. This answers *how would this model do on a different set of patients*,
rather than *how much would the score move if beats were re-drawn from these
same twenty-two*, which is not a useful question once those twenty-two have been
seen. Two thousand resamples are drawn and the 2.5th and 97.5th percentiles of
the resulting distribution are reported. A beat-level bootstrap is computed
alongside, solely for comparison.

Writing $Y_{ij}$ for the correctness of beat $j$ in recording $i$,

$$
Y_{ij} = \mu + a_i + \varepsilon_{ij}, \qquad
a_i \sim (0, \sigma_a^2), \quad \varepsilon_{ij} \sim (0, \sigma_\varepsilon^2)
$$

$$
\rho = \frac{\sigma_a^2}{\sigma_a^2 + \sigma_\varepsilon^2}, \qquad
D_{\text{eff}} = 1 + (\bar{m} - 1)\rho, \qquad
n_{\text{eff}} = \frac{n}{D_{\text{eff}}}
$$

with $\rho$ estimated by the method of moments from a one-way random-effects
model [Searle et al. 1992], using the effective group size for unbalanced
designs, and $D_{\text{eff}}$ the design effect of Kish [1965]. The
design-effect relation is derived for a **mean** and for equal group sizes;
§4.2 reports where it holds and where it does not.

These procedures are standard wherever observations are clustered within
patients, and are established practice in clinical and diagnostic evaluation.
They appear not to have reached this benchmark. The 2025 systematic review
summarises the reported performance of 122 studies, and that summary consists of
point estimates throughout -- sensitivity, positive predictivity, F1, accuracy --
with no interval accompanying any of them; where variation is reported it is
across training runs rather than across patients. No instance of the recording
being used as the resampling unit was found in the studies consulted here.

The bootstrap resamples rows of the stored prediction table. Nothing is
retrained and no models are combined, so it measures uncertainty from the
evaluation sample with the model held fixed. Uncertainty from training -- from
initialisation and from which records fall in validation -- is measured
separately by repeating seeds. The two are not combined; §5.7 notes the
consequence.

---

## 4. Results

### 4.1 The protocol gap

Macro F1 over N, S and V, seed 42, with 95% cluster bootstrap intervals:

| Protocol | point | 95% CI | $\rho$ | $n_{\text{eff}}$ |
|---|---:|---|---:|---:|
| **intra** | **0.9632** | [0.9142, 0.9805] | 0.013 | 2,922 |
| **inter** | **0.5924** | [0.5305, 0.6264] | 0.346 | **64** |

The intervals are separated by 0.28. Over three seeds the four-class figures are
0.934 ± 0.009 and 0.450 ± 0.023, a gap of 0.485, and the gap holds at 0.462 in
the `wide187` arm. Identical model, identical preprocessing, identical
augmentation; only the assignment of beats to halves differs.

### 4.2 The recording is the sampling unit

Under `inter`, correctness is clustered at $\rho = 0.352$ with 2,257 beats per
recording, giving $D_{\text{eff}} = 795$ and $n_{\text{eff}} = 62$ from 49,660
beats. Fifty thousand beats carry the information of sixty-two independent
observations.

| Statistic | beat bootstrap | cluster bootstrap | ratio | $\sqrt{D_{\text{eff}}}$ |
|---|---:|---:|---:|---:|
| **accuracy** | 0.00457 | 0.11097 | **24.27** | **28.20** |
| macro F1 (N/S/V) | 0.0110 | 0.1761 | 16.04 | 28.20 |

The design-effect relation is derived for a mean, and accuracy is a mean: there
the prediction and the observation agree to within 16%. Applied to macro F1 --
a ratio of ratios, outside the derivation -- it misses by 76%. The validation is
the accuracy row; the macro F1 ratio is reported as a descriptive quantity.

The residual 16% is in the expected direction. Group sizes are unequal, ranging
from 1,517 to 3,361 beats, which the formula does not accommodate; and a
non-parametric cluster bootstrap over a small number of groups is generally held
to understate variance, though the magnitude under these conditions has not been
checked against a source.

> ⚠️ *Not sure:* the small-cluster property is widely cited but its magnitude
> under these conditions has not been verified against a source here.

**What this establishes is the resampling unit, not the size of the gap.** The
gap is established in §4.1 and survives it.

### 4.3 Per-class decomposition

Seed 42, `narrow`:

| Class | intra F1 | inter F1 | penalty | $N_{\text{eff}}$ (DS1) |
|---|---:|---:|---:|---:|
| N | 0.996 | 0.964 | 0.032 | 21.01 |
| V | 0.983 | 0.813 | 0.169 | 7.15 |
| **S** | 0.911 | **0.000** | **0.911** | 4.48 |
| **F** | 0.807 | **0.000** | **0.807** | **1.24** |

Spearman rank correlation between $N_{\text{eff}}$ and penalty is -0.800, and
-1.000 in the `wide400` arm where the representation is least impoverished.
Four points cannot support a p-value; the ordering is the observation.

Two mechanisms produce it, and they are different. Ventricular beats survive on
morphology, having a wide QRS that does not depend on knowing the patient.
Supraventricular beats do not: their QRS is normal, they are defined by
prematurity, and with timing absent from the representation only
patient-specific morphology remains -- which a beat-level split supplies and a
record-level split does not. Fusion beats fail on both counts, being
intermediate in form and drawn from one patient.

### 4.4 Where the penalty sits

Accuracy across the 22 test recordings has a median of 0.977 and a range from
0.224 to 1.000. Six recordings exceed 0.99. **Three carry 67% of all errors.**

| Record | n | accuracy | note |
|---|---:|---:|---|
| **232** | 1,780 | **0.224** | 1,382 of its 1,780 beats (78%) are supraventricular |
| 213 | 3,250 | 0.826 | supplies 93% of DS2's fusion beats |
| 233 | 3,077 | 0.908 | |

The inter-patient penalty is not a uniform limit of the model. It is failure on
particular patients, and record 232 is a patient whose dominant rhythm has no
counterpart in the training half.

### 4.5 Capacity

Three models on identical inputs and splits, macro F1 over N/S/V:

| Model | capacity | intra | inter | gap | $\rho$ (inter) |
|---|---|---:|---:|---:|---:|
| **1-NN** | memorisation only | 0.9176 | 0.5219 | **0.396** | **0.418** |
| residual CNN | convolutional | 0.9632 | 0.5924 | 0.371 | 0.346 |
| logistic | linear | 0.6872 | 0.4727 | **0.215** | 0.293 |

The gap orders with capacity, and the linear model's is roughly half the
network's -- it has less to lose because it can memorise less.

The sharper observation is in the intra column: **storing the training set and
returning its nearest member reaches 0.918 against the network's 0.963.** Most
of what a beat-level split measures is retrievable similarity rather than
learned pattern, because the nearest neighbour of a test beat is generally
another beat from the same recording.

Intraclass correlation orders the same way. The model that only memorises has
the highest, at 0.418; the linear model the lowest. $\rho$ is behaving as an
index of how much a model depends on knowing the patient.

### 4.6 Representation

| Arm | N | S | V | F | macro (N/S/V), 3 seeds |
|---|---:|---:|---:|---:|---|
| `narrow` | 0.964 | 0.000 | 0.813 | 0.000 | 0.599 ± 0.030 |
| `wide187` | 0.967 | 0.090 | 0.895 | 0.000 | **0.654 ± 0.004** |
| `wide400` | 0.970 | **0.641** | 0.818 | 0.000 | 0.701 ± **0.094** |
| `rr_ratio` | 0.963 | 0.217 | 0.708 | 0.000 | 0.629 (1 seed) |
| `wide187_rr` | 0.970 | 0.064 | 0.924 | 0.000 | 0.652 (1 seed) |

Crossing window context against explicit intervals gives an interaction of
**-0.026**: the two sources of timing are redundant, not complementary. Adding
interval features to the wider window buys 0.0015 over the window alone.

The per-class view shows substitution rather than plateau. Summing the two
ectopic F1 scores gives 0.813, 0.985, 0.925 and 0.988 across the four arms --
the split between the classes moves freely while the total does not.
Prematurity is a property the two share, at normalised ratios of 0.763 and
0.733, so a network handed timing learns that a beat is ectopic without learning
which kind, and assigns it to whichever is four times more common in training.

Published inter-patient results give the scale. de Chazal et al. report a
supraventricular sensitivity of 75.9% at a positive predictivity of 38.5%, and a
ventricular sensitivity of 77.7% at a positive predictivity of 81.9%. Among the
inter-patient studies tabulated in the 2025 systematic review, reported
supraventricular F1 scores run from about 0.62 to 0.77, with ventricular F1
around 0.83 to 0.90; a random-forest study that, like this one, drops the sparse
classes from the classification reports 0.980, 0.731 and 0.909 for normal,
supraventricular and ventricular beats. Fusion, where reported at all, fares
badly: one inter-patient study lists a sensitivity of 0.25% for it.

Against those, normal and ventricular performance here is comparable --
ventricular sensitivity exceeds de Chazal's in every wide arm at a similar
positive predictivity -- and supraventricular performance is not. The best arm
reaches 0.641 in one seed and 0.26 averaged over three, against a published
range beginning around 0.62.

The comparison should be read with its differences in view. Both of those
classifiers use two leads, explicit morphological features and multi-scale RR
intervals, where this one uses a single lead and, in most arms, no interval
features at all.

**The `wide400` result does not survive its seeds.** Supraventricular F1 reads
0.641, 0.071 and 0.075 across three; the standard deviation of the arm's macro
F1 is 25 times that of `wide187`. In the seed that succeeded, record 232 --
which supplies 75% of DS2's supraventricular beats -- was classified at 0.889
against 0.000 elsewhere. The apparent gain is one recording landing correctly,
and it landed once in three.

### 4.7 Rebalancing

| $\beta$ | weight ratio | train acc | macro F1 (4-class) |
|---:|---:|---:|---:|
| 0 | 1.0 | 0.977 | **0.450** |
| 0.5 | 9.9 | 0.971 | 0.449 |
| 0.75 | 31.3 | 0.884 | 0.391 |
| **1.0** | **98.4** | **0.118** | **0.220** |

At $\beta = 1$ -- plain inverse frequency -- training collapses within an epoch.
Equalising the total loss contribution of every class means 414 fusion beats
carry the weight of 40,753 normal ones, and since those 414 come from
essentially one patient the network cannot learn to discriminate them.
Predicting the rare classes indiscriminately becomes the faster way to reduce
the weighted loss. No exponent improves on zero.

How much rebalancing a dataset tolerates is itself a measurement of how deeply
its classes are nested, and is reported here rather than tuned away.

### 4.8 Two refuted hypotheses

**Post-hoc prior correction.** The tilt introduced by rebalancing is a known
quantity and can be undone at prediction time:
$p_\beta(c \mid x) \propto p_{\text{model}}(c \mid x) / \tilde{\pi}_c^{\beta}$.
The hypothesis, stated before the test, was that the oversampler's advantage was
a shift of the decision threshold, so applying this to the unweighted model's
stored probabilities should recover most of it.

It does not. Macro F1 decreases monotonically from 0.444 at $\beta = 0$ to 0.225
at $\beta = 1.25$, and supraventricular F1 never exceeds 0.084. Prior correction
is a monotone transformation of the scores and can only redistribute decisions
along an existing ranking; in the unweighted model the supraventricular score is
almost always below the normal score, and no threshold recovers a separation
that is not there.

**Oversampling raises the ceiling.** The natural reading of that result is that
oversampling changed the ranking itself. One run supported it: under `wide187`,
seed 42, the sampler reached 0.671 against 0.650 unweighted. Two further seeds
removed the effect.

| | n | macro F1 (N/S/V) | seeds |
|---|---:|---|---|
| unweighted | 3 | **0.6544 ± 0.0037** | 0.6505, 0.6548, 0.6579 |
| oversampled | 3 | 0.6498 ± **0.0272** | 0.6705, 0.6190, 0.6598 |

The mean is marginally lower and the standard deviation **7.3 times larger**.
Seed 42 was the best of three draws from a much noisier distribution. The
nesting structure predicts this: oversampling a class replicates the few
recordings that supply it, so a run's outcome comes to depend on which of those
survived the validation split. The procedure does not add patient diversity; it
amplifies dependence on the diversity already there.

Pairing the best representation with the best procedure compounds nothing:
`wide187_rr` with oversampling reaches 0.578, below every other configuration
including the baseline, with normal recall falling to 0.797. Both operations
push toward the minority classes and applied together they overshoot.

### 4.9 Calibration

Aggregate calibration under `inter` looks unremarkable: mean confidence 0.9343
against accuracy 0.9333, an expected calibration error of 0.024. Per recording
it does not.

| Record | n | confidence | accuracy | overconfidence | ECE |
|---|---:|---:|---:|---:|---:|
| **232** | 1,780 | **0.949** | **0.224** | **+0.725** | **0.725** |
| 213 | 3,250 | 0.952 | 0.826 | +0.127 | 0.127 |
| 231 | 1,570 | 0.788 | 0.951 | -0.163 | 0.163 |

**The model is most confident where it is least right.** Twenty of the
twenty-two recordings are mildly underconfident; record 232 is overconfident by
0.725 and holds only 3.6% of the beats, so the aggregate absorbs it entirely.
Low accuracy invites review. Misplaced confidence does not.

**The observation set.** The 15 unclassifiable beats, never seen in training,
were answered as normal ten times and ventricular five, at a median confidence
of 0.895, six of them above 0.9. Four of the five ventricular calls are
transients from record 105, answered at 0.875 to 0.954. Fifteen beats support no
estimate and this is reported as an observation, but a classifier that has only
ever seen clean morphology is not hedging when it meets artefact.

### 4.10 Learning rate

The inter-patient runs peak early -- a mean best epoch of 8.9 against 36.7 for
intra-patient -- which invites the objection that they are undertrained.

| Learning rate | best epoch | **train accuracy there** | macro (N/S/V) |
|---:|---:|---:|---:|
| 3e-4 | 21 | **0.9951** | 0.660 |
| 1e-3 | 16 | **0.9960** | 0.650 |
| 3e-3 | 11 | **0.9944** | 0.673 |

Training accuracy exceeds 99.4% at the validation peak under every setting, so
the network is not short of fitting; the peak moves by a factor of two while the
score moves by 0.023. The early peak is the point at which generalisable signal
in the training recordings runs out. Against ourselves: 3e-3 scores highest, so
the 1e-3 default is not demonstrably optimal on this single seed.

### 4.11 What the intervals permit

Every paired comparison between configurations contains zero.

| Comparison | difference | 95% CI |
|---|---:|---|
| `wide187` - `narrow` | +0.067 | [-0.007, +0.159] |
| `wide400` - `wide187` | +0.107 | [-0.072, +0.263] |
| oversampled - unweighted | (3 seeds) | means differ by 0.005, sd 7x |

The supraventricular interval under `wide400` runs from 0.097 to 0.827, a width
of 0.73 against a point estimate of 0.641, because record 232 supplies 75% of
that class in DS2 and is absent from 37% of resamples. The point estimate
reports that one patient was classified correctly.

Note the contrast with the seed-level figures: a standard deviation of 0.004
across seeds in the `wide187` arm against a record-level interval width of 0.171
in the same arm. Seed variation measures whether the optimisation is
reproducible; record variation measures whether the result is about the model or
about which patients were evaluated. The second is forty times the first, and
the convention of reporting several seeds addresses only the first.

### 4.13 The supraventricular class is not homogeneous in timing

The interval features assume that a supraventricular beat is early relative to
its patient's own rhythm. Measured per recording, that assumption holds in some
and not in others.

| Record | S beats | S share of rhythm | **S pre-RR / local mean** | N pre-RR / local mean |
|---|---:|---:|---:|---:|
| 201 | 128 | 6.5% | **0.499** | 1.045 |
| 202 | 55 | 2.6% | 0.577 | 0.999 |
| 222 | 209 | 8.4% | 0.637 | 1.007 |
| 220 | 94 | 4.6% | 0.652 | 1.004 |
| 100 | 33 | 1.5% | 0.747 | 1.000 |
| **232** | **1,382** | **77.6%** | 0.736 | **1.812** |
| 209 | 383 | 12.8% | 0.930 | 1.008 |
| 234 | 50 | 1.8% | 0.986 | 0.999 |
| 124 | 31 | 1.9% | 0.996 | 1.001 |
| **207** | 106 | 5.7% | **1.003** | 1.001 |

**Normalised prematurity for the supraventricular class ranges from 0.499 to
1.003 across recordings.** In records 207, 124 and 234 these beats are not early
at all. The AAMI grouping collects atrial premature, aberrated atrial premature,
nodal premature and supraventricular premature beats under one label, and not
every member of that set is defined by timing. A feature built on prematurity
therefore describes some of the class and not the rest, which is part of why the
interval arm in §4.6 gained less than its motivation suggests.

**A second observation concerns the normalisation itself.** Record 232 is the
only recording where normal beats do not read near 1.0; they read 1.812. Its
rhythm is 77.6% supraventricular, so the eleven-beat local mean is largely made
of ectopic intervals and the normal beats become the outliers against it.
Normalising by a local mean assumes ectopy is the exception; where it is the
rule, the reference frame is built from the thing it is meant to contrast
against.

This did not cost accuracy. Under `wide400` the model reaches 0.985 recall on
that recording's normal beats despite their unusual ratio, which says it is not
relying on interval features alone. What it does cost is the argument of §5.6:
normalisation is patient-invariant only where the quantity being normalised
away is not itself the pathology.

The errors on record 232 fall as the rest of the study predicts. Of 1,382
supraventricular beats, 1,108 are correct, 180 are called ventricular and 94
normal -- two thirds of the errors going to the other ectopic class, in a
recording containing no ventricular beats at all.

### 4.12 Sensitivity: records 201 and 202

Records 201 and 202 come from the same male subject, and de Chazal places 201 in
DS1 and 202 in DS2, so the published partition is not strictly subject-disjoint.
Excluding 202 changes macro F1 from 0.4443 to 0.4442. The caveat is real; its
effect is negligible.

---

## 5. Discussion

### 5.1 One structure

Class imbalance and patient heterogeneity are usually treated as separate
problems. In this database they are one: minority-class beats are unevenly
nested inside recordings, so any operation on the class axis is also an
operation on the patient axis. Oversampling a class replicates a patient.
Reweighting a loss reweights a patient. $N_{\text{eff}}$ makes the coupling
explicit, and it is what explains why the class ordering of the penalty is not
the class ordering of the frequencies.

The four sections below are four consequences of that one structure rather than
four separate findings.

### 5.2 The structure inflates performance

Separating patients costs 0.485 in macro F1, and the intervals do not overlap.
The mechanism is visible in the capacity axis: a nearest-neighbour classifier
reaches 0.918 under a beat-level split against the network's 0.963, so most of
what that setting measures is retrievable rather than learned. It is visible
again in the optimisation trace, where the inter-patient runs peak at epoch 9
and the intra-patient runs continue improving past 40 -- continuing to fit the
training recordings is rewarded when those recordings also furnish the test
beats, and paid for when they do not.

### 5.3 The structure conceals itself

This is the finding least likely to be noticed from inside a beat-level
evaluation, and it operates at three levels.

**The clustering.** Intraclass correlation reads 0.013 under `intra` and 0.346
under `inter`. Measured in the first setting the data look very nearly
independent, and a beat-level bootstrap looks justified. Only after separating
patients does the clustering appear.

**The uncertainty.** Beat-level intervals are 24 times too narrow on accuracy,
a factor the design effect predicts to within 16%.

**The failure.** Aggregate expected calibration error is 0.024 while the worst
recording's is 0.725. The aggregate averages over patients and no patient
experiences an average.

Each level would, taken alone, license the conclusion that nothing is wrong.

### 5.4 The structure resists correction

Five rebalancing strengths, an oversampler, four representation arms, three
augmentation modes and three learning rates were tried. None improved on the
simplest configuration by more than the noise, and two -- inverse-frequency
weighting and the combination of the strongest representation with the strongest
procedure -- made things substantially worse.

The representation result is the most informative. Prematurity is shared between
the two ectopic classes at 0.763 and 0.733 of the local mean, so timing
identifies that a beat is ectopic without identifying which kind. Every arm
therefore trades one class against the other, with the sum of their F1 scores
confined between 0.81 and 0.99.

### 5.5 The structure prevents verification

Every paired comparison between configurations contains zero. With
$n_{\text{eff}} = 64$, twenty-two recordings do not distinguish these
interventions. This is not a statement that nothing helps; it is a statement
that this dataset cannot show whether something does.

The `wide400` arm makes the point concretely. Its supraventricular F1 of 0.641
in one seed is 0.07 in two others, and the interval on that figure runs from
0.097 to 0.827 because the class is three-quarters one recording. A single seed
and a beat-level interval would have reported a substantial improvement with
apparent confidence, and both were tried here before the additional seeds and
the cluster bootstrap contradicted them.

### 5.6 Absolute and normalised scales

Four axes of this project independently reached the same conclusion.

| Axis | Absolute, patient-dependent | Normalised, patient-invariant |
|---|---|---|
| **Amplitude** | raw millivolts | divided by the recording's IQR |
| **Sample size** | beat count | $N_{\text{eff}}$ |
| **Timing** | pre-RR in seconds | pre-RR over the local mean |
| **Context** | a fixed-duration window | *no equivalent* |

The fourth row is where the principle bites. A window of fixed duration carries
timing only on an absolute scale, because how many neighbouring beats fall
inside it is the patient's resting rate. There is no way to normalise a window.

There is a limit to it, and §4.13 marks it: normalising by a local mean assumes
that what is being normalised away is nuisance rather than pathology. In record
232, where three-quarters of the rhythm is ectopic, the local mean is built from
the ectopy and the normal beats become the outliers. Ratios are patient-
invariant when the patient's baseline is a baseline.

Fusion beats show what the absolute scale costs. On an absolute scale they look
premature --
a median preceding interval of 0.564 s against 0.768 s for normal beats -- but
normalised they sit at 0.983. Records 208 and 213 supply almost all of them and
run at 103 and 108 beats per minute against 75 and 69 for typical records. The
fusion beats are not early; the patients who have them are fast. A model given
absolute timing would learn to detect fusion as *a patient whose heart rate is
high*, and would appear to transfer from DS1 to DS2 only because those two
recordings happen to share the trait.

### 5.7 Limitations

Three kinds, and conflating them would misrepresent the work.

**Design choices, reversible.** Supraventricular F1 reaches 0.26 over three
seeds against a published inter-patient range beginning around 0.62, and the
following choices account for the difference. The morphology representation is a single lead
with no explicit features -- no QRS duration, no P-wave shape -- and no
band-pass filtering, so the low-amplitude P wave that distinguishes an atrial
ectopic beat is present in the window but weak. Interval features come from one
local window where the literature uses several scales. No rebalancing is
applied, because none improved the aggregate. Each could be changed.

**Structural, not reversible.** The database holds 47 subjects and one recording
each, so subject, session and electrode placement are perfectly confounded and
their contributions cannot be separated even in principle. $N_{\text{eff}}$ is
1.24 for fusion. No amount of compute changes either number: parameters can be
added, patients cannot.

**Properties of the modality, not losses.** A single-lead recording is one
potential against time. It contains no spatial or inter-lead phase information,
and none was discarded in producing this representation; obtaining it would
require a multi-lead recording. This is stated because the first two categories
are easy to read as evidence that a richer signal was thrown away.

Two further caveats. Three of the five representation arms rest on one seed
each, and the `wide400` result shows what one seed can hide. And uncertainty
from training and from evaluation are measured separately rather than combined;
since the seed-level standard deviation is forty times smaller than the
record-level interval, combining them would not change any conclusion, but the
intervals reported are evaluation-only.

---

## 6. Pending

No further training is required; every item below reads the stored prediction
tables.

- Figures for all sections, including exploratory figures for the wide-window
  arms, which have so far been inspected only for `narrow`.
- Per-record calibration figure and the observation-set panel.
- An interval on the arm comparison for the two single-seed arms.

## 7. Future work

- **A hierarchical model with the recording as a random effect** would separate
  the variance components of class and patient effects, formalising the
  observation that oversampling does not increase effective patient diversity.
  Partial pooling would shrink the estimate for a class supported by one patient
  and widen its uncertainty automatically, which is what loss weighting
  attempted here by an indirect route and failed at.
- **A factorial design separating prior rebalancing from invariance injection**,
  which the class-dependent multipliers of the present augmentation confound.
- **Both timing scales supplied together.** Absolute intervals were excluded
  because they encode resting rate as much as prematurity, but the normalised
  ratios may not capture everything; supplying both would test whether the
  additional information outweighs the memorisation route it opens. Note that
  §5.5 applies -- this dataset could not adjudicate it.
- **Beat transitions as the unit of classification.** An atrial ectopic beat
  appears in context as an N-to-S transition, so classifying transitions would
  build prematurity into the label definition. It changes the task and would be
  a separate study.
- **Extension to INCART or the MIT-BIH Supraventricular database**, to test
  whether the nesting structure is a property of this database or of ambulatory
  arrhythmia data generally. This is the single most informative addition
  available, because it addresses the limitation that no method can.
- Generative augmentation is **not** on this list. A model fitted to one
  patient's fusion beats produces that patient's fusion beats;
  $N_{\text{eff}}$ is unchanged. Information that is absent cannot be
  synthesised.

---

## Appendix A. Superseded results

The previous version used the public preprocessed CSV release and reported
0.9746 accuracy and 0.8763 macro F1 over five classes, with materialized
augmentation, on a beat-level split.

Those results are **not comparable** and are excluded from every comparison in
§4. They differ in four respects simultaneously: data source, split protocol,
class definition (five classes including roughly eight thousand paced beats),
and beat extraction with min-max scaling.

One objection deserves an answer: perhaps the new preprocessing is what makes
patient-level evaluation hard. Two things argue against it. The same pipeline
reaches a four-class macro F1 of 0.924 under a beat-level split, not low against
the CSV-based 0.876 over five classes. And the protocol gap holds between 0.46
and 0.49 across four representation arms differing substantially in window,
sampling rate and features. The gap is not an artefact of any one preprocessing
choice.

The code and artefacts that produced the earlier results are preserved at the
git tag `v0.1.0-csv`.

---

## References

1. Kachuee M, Fazeli S, Sarrafzadeh M. *ECG Heartbeat Classification: A Deep
   Transferable Representation.* IEEE ICHI-W, 2018. arXiv:1805.00794.
2. de Chazal P, O'Dwyer M, Reilly RB. *Automatic Classification of Heartbeats
   Using ECG Morphology and Heartbeat Interval Features.* IEEE Trans Biomed Eng.
   2004;51(7):1196-1206.
3. Moody GB, Mark RG. *The Impact of the MIT-BIH Arrhythmia Database.* IEEE Eng
   Med Biol Mag. 2001;20(3):45-50.
4. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.
   2000;101(23):e215-e220.
5. MIT-BIH Arrhythmia Database. PhysioNet. https://physionet.org/content/mitdb/1.0.0/
6. ANSI/AAMI EC57. *Testing and Reporting Performance Results of Cardiac Rhythm
   and ST Segment Measurement Algorithms.* 1998.
7. Luz EJS, Schwartz WR, Camara-Chavez G, Menotti D. *ECG-based Heartbeat
   Classification for Arrhythmia Detection: A Survey.* Comput Methods Programs
   Biomed. 2016;127:144-164.
8. *Investigating Feature Selection and Random Forests for Inter-Patient
   Heartbeat Classification.* Algorithms. 2020;13(4):75. *(reported F1 scores
   for N, SVEB and VEB; verify author list before submission)*
9. Silva GAL, Silva PHL, Moreira GJP, Freitas VLS, Gertrudes JC, Luz EJS.
   *A Systematic Review of ECG Arrhythmia Classification: Adherence to
   Standards, Fair Evaluation, and Embedded Feasibility.* arXiv:2503.07276,
   2025.
10. Huang H, Liu J, Zhu Q, Wang R, Hu G. *A New Hierarchical Method for
   Inter-Patient Heartbeat Classification Using Random Projections and RR
   Intervals.* BioMedical Engineering OnLine. 2014;13:90.
11. Kish L. *Survey Sampling.* Wiley, 1965. (design effect)
12. Searle SR, Casella G, McCulloch CE. *Variance Components.* Wiley, 1992.
   (unbalanced one-way random effects)
13. Field CA, Welsh AH. *Bootstrapping Clustered Data.* J R Stat Soc B.
    2007;69(3):369-390.
14. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap.* Chapman & Hall,
    1993.
15. Elkan C. *The Foundations of Cost-Sensitive Learning.* IJCAI, 2001.
16. *DeepArrhythmia: Segment-Contextualized ECG Arrhythmia Classification via
    Selective Evidence Acquisition.* arXiv:2605.16441. *(cited for fusion-class
    performance; verify against the source before submission)*