"""On-the-fly augmentation for heartbeat vectors.

The reference paper does not describe an augmentation procedure, so none of
this reproduces published work; the design is our own and is documented here
rather than inferred from an artefact.

A note on why this file changed. The previous version clipped its output to
[0, 1], which was correct while beats were min-max normalised inside a ten
second window. Under the current representation -- per-beat median subtraction
followed by a record-level robust scale -- beats span roughly -2 to +8, and
that same clip flattened every R peak to 1.0 and every S trough to 0.0. It
raised no error; it simply removed the QRS complex from the augmented copies.

The lesson generalises beyond this bug: an augmentation procedure cannot be
specified independently of the representation it acts on. Where the source of
a representation is undocumented, the augmentation built on top of it is not
reproducible either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecg_classification.constants import R_PEAK_INDEX, SAMPLE_LENGTH


def _resize_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a 1D signal to a new length by linear interpolation.

    Linear interpolation is used rather than FFT resampling because an isolated
    beat is not periodic: its first and last samples sit on the baseline but
    are not equal, and FFT resampling answers that discontinuity with ringing
    around the sharpest feature in the signal, which here is the QRS complex.
    """
    source = np.linspace(0.0, 1.0, num=len(signal), dtype=np.float64)
    target = np.linspace(0.0, 1.0, num=int(target_length), dtype=np.float64)
    return np.interp(target, source, signal).astype(np.float32)


def time_stretch(
    signal: np.ndarray,
    rng: np.random.Generator,
    sample_length: int = SAMPLE_LENGTH,
    r_peak_index: int = R_PEAK_INDEX,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    """Randomly stretch or compress the signal in time, holding the R peak fixed.

    Every beat in this dataset is cut with its R peak at a known sample, and
    the network sees a fixed window because of it. That sample is
    ``R_PEAK_INDEX``, which accounts for any resampling between the extraction
    window and the model input, so this holds in every representation arm. Naive stretching breaks that
    invariant: resampling to a different length moves the peak proportionally,
    so a compressed beat drifts left and a stretched beat drifts right. The
    network would then be asked to absorb a positional shift that the
    segmentation was designed to eliminate.

    Here the beat is resampled, the peak's new position is computed, and the
    window is re-cut around it so the peak returns to ``r_peak_index``.

    Where the re-cut window runs past the resampled signal, the edge value is
    repeated rather than zero-padded. Stretching in time represents a slower
    heart rate, whose visible effect is a longer baseline either side of the
    complex; continuing the baseline expresses that, whereas a constant fill
    would introduce a step that no recording contains.
    """
    scale = rng.uniform(min_scale, max_scale)
    stretched_length = max(8, int(round(sample_length * scale)))

    stretched = _resize_signal(signal, stretched_length)
    if stretched_length == sample_length:
        return stretched

    # Where the peak landed after resampling, and where we need it to be.
    moved_peak = int(round(r_peak_index * stretched_length / sample_length))
    start = moved_peak - r_peak_index
    stop = start + sample_length

    pad_before = max(0, -start)
    pad_after = max(0, stop - stretched_length)
    if pad_before or pad_after:
        stretched = np.pad(stretched, (pad_before, pad_after), mode="edge")
        start += pad_before

    return stretched[start : start + sample_length].astype(np.float32)


def amplitude_scale(
    signal: np.ndarray,
    rng: np.random.Generator,
    min_scale: float = 0.85,
    max_scale: float = 1.15,
) -> np.ndarray:
    """Apply a simple amplitude scaling factor.

    Beats arrive with their own median already subtracted, so the baseline sits
    near zero and scaling changes the size of the deflections without shifting
    the isoelectric line -- which is what varying recording amplitude does, and
    is not what this operation did under the previous representation.
    """
    scale = rng.uniform(min_scale, max_scale)
    return (signal * scale).astype(np.float32)


def add_gaussian_noise(
    signal: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.05,
) -> np.ndarray:
    """Inject a small amount of Gaussian noise.

    The default is expressed in units of the record's interquartile range,
    which the normalisation sets to one. At 0.05 the perturbation is a few
    percent of a typical QRS deflection, comparable to the muscle artefact
    present in ambulatory recordings.

    The previous default of 0.01 was chosen against a [0, 1] representation,
    where it was one percent of full scale. Carried over unchanged it would sit
    near a tenth of a percent here and do nothing at all.
    """
    noise = rng.normal(0.0, noise_std, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


@dataclass(slots=True)
class BeatAugmenter:
    """On-the-fly augmentation for heartbeat vectors.

    Amplitude and time scaling are not arbitrary choices. Between-patient
    differences in this database are driven by thoracic impedance, cardiac axis
    and recorder gain on the amplitude side, and by heart rate on the time
    side, so the two operations act on the axes along which recordings actually
    differ from one another.

    That said, the pairing is a motivation and not a model. Uniform time
    scaling stretches the QRS complex along with everything else, whereas real
    rate variation compresses the diastolic interval and the QT segment while
    leaving QRS duration largely alone. And because this database holds one
    recording per subject, subject, session and electrode placement cannot be
    told apart at all. What follows is an empirical invariance, and it is
    described that way in the report.

    Clipping is disabled by default. Setting bounds only makes sense for a
    representation with a known range, and this one has none.
    """

    sample_length: int = SAMPLE_LENGTH
    r_peak_index: int = R_PEAK_INDEX
    stretch_probability: float = 0.50
    amplitude_probability: float = 0.50
    noise_probability: float = 0.30
    noise_std: float = 0.05
    clip_min: float | None = None
    clip_max: float | None = None

    def __call__(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        augmented = signal.astype(np.float32, copy=True)

        if rng.random() < self.stretch_probability:
            augmented = time_stretch(
                augmented,
                rng=rng,
                sample_length=self.sample_length,
                r_peak_index=self.r_peak_index,
            )

        if rng.random() < self.amplitude_probability:
            augmented = amplitude_scale(augmented, rng=rng)

        if rng.random() < self.noise_probability:
            augmented = add_gaussian_noise(augmented, rng=rng, noise_std=self.noise_std)

        if self.clip_min is None and self.clip_max is None:
            return augmented.astype(np.float32)

        return np.clip(augmented, self.clip_min, self.clip_max).astype(np.float32)