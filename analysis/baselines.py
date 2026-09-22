"""Capacity baselines: where the protocol gap comes from.

The central result is that separating patients costs about half the macro F1.
The explanation offered for it is memorisation -- that a beat-level split lets a
network recognise the patient rather than the arrhythmia. That explanation makes
a prediction, and this module tests it.

If the gap is memorisation, it should scale with how much a model *can*
memorise. Three models spanning that axis are run on identical inputs and
identical splits:

**1-nearest neighbour** stores the training set and answers with its closest
member. It memorises and does nothing else, so it is the upper extreme. Under a
beat-level split its neighbours include other beats from the same recording;
under a record-level split they cannot.

**Multinomial logistic regression** fits one linear boundary per class. It has
almost no capacity to memorise, and its coefficients live on the 187 sample
positions, so they can be read directly against the waveform. Note also that an
L2-regularised fit is the MAP estimate under a Gaussian prior with precision
equal to the penalty, so the capacity axis is a prior-strength axis as well.

**The residual CNN** sits between them and is read from its existing run.

The prediction is that the gap widens with capacity: near-total for 1-NN,
smallest for the linear model. Neither baseline is offered as a competitor to
the network; both are instruments for measuring what the network's advantage is
made of.

Usage
-----
    python -m analysis.baselines --protocol inter
    python -m analysis.baselines --protocol inter --protocol intra
    ECG_REPRESENTATION=wide187 python -m analysis.baselines --protocol inter --protocol intra
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_classification.constants import CLASS_SYMBOLS, NUM_CLASSES, REPRESENTATION
from ecg_classification.data import build_dataset_bundle

SELECTION = (0, 1, 2)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    flat = np.bincount(y_true * NUM_CLASSES + y_pred, minlength=NUM_CLASSES**2)
    return flat.reshape(NUM_CLASSES, NUM_CLASSES).astype(np.float64)


def per_class_f1(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 2.0 * np.diag(matrix) / denominator, 0.0)


def flatten(features: np.ndarray, intervals: np.ndarray | None) -> np.ndarray:
    """Both baselines take a flat vector; interval features are appended.

    The CNN receives the intervals as constant planes and separates them again
    at its classifier. Appending them to a flat vector is the same information
    presented the way these models expect it, which keeps the comparison about
    capacity rather than about plumbing.
    """
    flat = np.asarray(features, dtype=np.float32)
    if intervals is None:
        return flat
    return np.concatenate([flat, np.asarray(intervals, dtype=np.float32)], axis=1)


def fit_logistic(x_train, y_train, x_test, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    # multi_class was removed in scikit-learn 1.7; multinomial is the default
    # for more than two classes, so dropping it changes nothing but the version
    # this file runs under.
    model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    scaled_test = scaler.transform(x_test)
    return model.predict(scaled_test), model.coef_


def fit_nearest_neighbour(x_train, y_train, x_test) -> np.ndarray:
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, records: np.ndarray) -> dict:
    matrix = confusion(y_true, y_pred)
    f1 = per_class_f1(matrix)
    return {
        "accuracy": float(np.diag(matrix).sum() / matrix.sum()),
        "macro4": float(f1.mean()),
        "macro3": float(f1[list(SELECTION)].mean()),
        "f1": f1,
        "confusion": matrix,
        "y_pred": y_pred,
        "record_id": records,
        "y_true": y_true,
    }


def run_protocol(protocol: str, seed: int, validation_size: float, subsample: int | None) -> dict:
    bundle = build_dataset_bundle(protocol=protocol, seed=seed, validation_size=validation_size)

    # Validation is not needed: neither baseline early-stops. Folding it back
    # into training keeps the linear model on the same beats the network saw in
    # total, so the comparison is not confounded by training-set size.
    x_train = np.concatenate([
        flatten(bundle.X_train, bundle.i_train),
        flatten(bundle.X_valid, bundle.i_valid),
    ])
    y_train = np.concatenate([bundle.y_train, bundle.y_valid])
    x_test = flatten(bundle.X_test, bundle.i_test)
    y_test = bundle.y_test

    print(f"\n  {protocol}: {len(x_train):,} train beats, {len(x_test):,} test, "
          f"{x_train.shape[1]} features")

    results = {}

    start = time.time()
    predicted, coefficients = fit_logistic(x_train, y_train, x_test, seed)
    results["logistic"] = evaluate(y_test, predicted, bundle.r_test)
    results["logistic"]["coefficients"] = coefficients
    print(f"    logistic  {time.time() - start:6.1f}s   "
          f"macro3 {results['logistic']['macro3']:.4f}")

    # 1-NN over fifty thousand training beats and fifty thousand queries is the
    # slow step. Subsampling the reference set trades a little accuracy for
    # tractability; the point of this arm is the shape of the gap, not its last
    # decimal.
    if subsample is not None and len(x_train) > subsample:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(x_train), subsample, replace=False)
        x_reference, y_reference = x_train[keep], y_train[keep]
        print(f"    (1-NN reference set subsampled to {subsample:,})")
    else:
        x_reference, y_reference = x_train, y_train

    start = time.time()
    predicted = fit_nearest_neighbour(x_reference, y_reference, x_test)
    results["1nn"] = evaluate(y_test, predicted, bundle.r_test)
    print(f"    1-NN      {time.time() - start:6.1f}s   "
          f"macro3 {results['1nn']['macro3']:.4f}")

    return results


def load_cnn(protocol: str, seed: int) -> dict | None:
    """Read the matching network run rather than retraining it."""
    prefix = "" if REPRESENTATION == "narrow" else f"{REPRESENTATION}-"
    directory = PROJECT_ROOT / "outputs" / f"{prefix}{protocol}-none-lossweight0-seed{seed}"
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None
    matrix = np.asarray(
        json.loads(metrics_path.read_text(encoding="utf-8"))["confusion_matrix"], dtype=np.float64
    )
    f1 = per_class_f1(matrix)
    return {
        "accuracy": float(np.diag(matrix).sum() / matrix.sum()),
        "macro4": float(f1.mean()),
        "macro3": float(f1[list(SELECTION)].mean()),
        "f1": f1,
        "confusion": matrix,
    }


def report(all_results: dict[str, dict[str, dict]]) -> None:
    order = ["1nn", "logistic", "cnn"]
    names = {"1nn": "1-NN", "logistic": "logistic", "cnn": "residual CNN"}
    capacity = {"1nn": "memorisation only", "logistic": "linear", "cnn": "convolutional"}

    print("\n" + "=" * 80)
    print("CAPACITY AND THE PROTOCOL GAP  (macro F1 over N/S/V)")
    print("=" * 80)
    print(f"  {'model':<16}{'capacity':<20}{'intra':>9}{'inter':>9}{'gap':>9}")
    print("  " + "-" * 63)

    for key in order:
        cells = {p: r.get(key) for p, r in all_results.items()}
        if not any(cells.values()):
            continue
        intra = cells.get("intra", {}).get("macro3") if cells.get("intra") else None
        inter = cells.get("inter", {}).get("macro3") if cells.get("inter") else None
        gap = f"{intra - inter:>9.4f}" if (intra is not None and inter is not None) else f"{'--':>9}"
        print(f"  {names[key]:<16}{capacity[key]:<20}"
              f"{intra if intra is None else f'{intra:.4f}':>9}"
              f"{inter if inter is None else f'{inter:.4f}':>9}{gap}")

    print("\n  The prediction was that the gap widens with capacity: a model that can")
    print("  only memorise loses everything when the patients change, a linear model")
    print("  has little to lose, and the network sits between them.")

    for protocol, results in all_results.items():
        print(f"\n  per class, {protocol}")
        print(f"    {'model':<16}" + "".join(f"{s:>9}" for s in CLASS_SYMBOLS))
        for key in order:
            if key not in results or results[key] is None:
                continue
            print(f"    {names[key]:<16}" + "".join(f"{v:>9.3f}" for v in results[key]["f1"]))


def save_predictions(results: dict, output_dir: Path, protocol: str) -> None:
    for key in ("logistic", "1nn"):
        if key not in results:
            continue
        directory = output_dir / f"{REPRESENTATION}-{protocol}-{key}"
        directory.mkdir(parents=True, exist_ok=True)
        result = results[key]
        pd.DataFrame({
            "record_id": result["record_id"],
            "beat_index": np.arange(len(result["y_true"])),
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
        }).to_csv(directory / "predictions.csv", index=False)
        json.dump(
            {
                "run_name": directory.name,
                "representation": REPRESENTATION,
                "protocol": protocol,
                "model": key,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro4"],
                "macro_f1_nsv": result["macro3"],
            },
            open(directory / "config.json", "w"),
            indent=2,
        )
        json.dump(
            {"macro_f1": result["macro4"], "confusion_matrix": result["confusion"].tolist()},
            open(directory / "metrics.json", "w"),
            indent=2,
        )
        print(f"  wrote {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", action="append", choices=["intra", "inter"],
                        help="repeat for both; defaults to both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--subsample", type=int, default=20000,
                        help="reference-set size for 1-NN; 0 disables subsampling")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    protocols = args.protocol or ["intra", "inter"]
    subsample = None if args.subsample == 0 else args.subsample

    print("=" * 80)
    print(f"BASELINES  representation={REPRESENTATION}  seed={args.seed}")
    print("=" * 80)

    all_results: dict[str, dict] = {}
    for protocol in protocols:
        results = run_protocol(protocol, args.seed, args.validation_size, subsample)
        cnn = load_cnn(protocol, args.seed)
        if cnn is None:
            print(f"    (no matching CNN run for {protocol}; its row will be blank)")
        else:
            results["cnn"] = cnn
        all_results[protocol] = results
        if not args.no_save:
            save_predictions(results, args.output_dir, protocol)

    report(all_results)


if __name__ == "__main__":
    main()