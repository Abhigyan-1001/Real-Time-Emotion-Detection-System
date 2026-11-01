"""Evaluate the trained model on a FER-2013 split.

Uses the SAME preprocessing and prediction code as the live app, so the numbers
describe the code that actually runs in the dashboard.

    python evaluate.py --split val           # development: also calibrates the threshold
    python evaluate.py --split test --final  # final, one-time report
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import keras
import matplotlib

matplotlib.use("Agg")  # write image files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

from src.metrics import classification_summary, reliability_table, threshold_sweep
from src.predictor import EmotionPredictor
from src.preprocessing import to_model_input

DATA_DIR = Path("data/fer2013")
RESULTS_DIR = Path("results")
MODEL_PATH = "models/emotion_cnn_v1.keras"
LABELS_PATH = "models/labels.json"
SEED = 42  # must match the training notebook
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def validation_paths(class_names):
    """Reproduce the exact 15% validation split used during training."""
    ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=(48, 48),
        batch_size=64,
        shuffle=True,
        seed=SEED,
        validation_split=0.15,
        subset="validation",
    )
    if list(ds.class_names) != list(class_names):
        raise SystemExit(f"Label order mismatch: {ds.class_names} vs {class_names}")
    paths = getattr(ds, "file_paths", None)
    if paths is None:
        raise SystemExit("This Keras version does not expose dataset.file_paths.")
    return [Path(p) for p in paths]


def test_paths(class_names):
    paths = []
    for name in class_names:
        folder = DATA_DIR / "test" / name
        paths += sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths


def read_gray(path, reader):
    """Read an image as a uint8 grayscale array.

    'tf' decodes like the training pipeline did; 'cv2' decodes with OpenCV.
    """
    if reader == "cv2":
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    data = tf.io.read_file(str(path))
    try:
        return tf.io.decode_image(data, channels=1, expand_animations=False).numpy()[:, :, 0].astype(np.uint8)
    except tf.errors.InvalidArgumentError:
        return None


def score_images(paths, predictor, reader="tf", batch_size=128):
    """Read, preprocess and score images. Returns (scores, kept_paths, skipped_count)."""
    scores, kept, skipped = [], [], 0
    batch, batch_paths = [], []

    def flush():
        if batch:
            scores.append(predictor.predict_scores(np.concatenate(batch, axis=0)))
            kept.extend(batch_paths)
            batch.clear()
            batch_paths.clear()

    for path in paths:
        image = read_gray(path, reader)
        model_input = None if image is None else to_model_input(image)
        if model_input is None:
            skipped += 1
            continue
        batch.append(model_input)
        batch_paths.append(path)
        if len(batch) == batch_size:
            flush()
    flush()
    return np.concatenate(scores, axis=0), kept, skipped


def print_report(summary):
    print(f"\nAccuracy : {summary['accuracy']:.3f}")
    print(f"Macro-F1 : {summary['macro_f1']:.3f}\n")
    print(f"{'class':<10}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for name, m in summary["per_class"].items():
        print(f"{name:<10}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{m['support']:>9d}")


def print_sweep(rows, class_names):
    print("\nThreshold sweep on single images (coverage = share of images that get a label)")
    header = " ".join(f"{n[:4]:>5}" for n in class_names)
    print(f"{'thr':>5}{'coverage':>10}{'acc|lab':>9}  {header}   <- coverage per TRUE class")
    for r in rows:
        acc = "-" if np.isnan(r["accuracy_on_accepted"]) else f"{r['accuracy_on_accepted']:.3f}"
        per_class = " ".join(f"{r['per_class_coverage'][n]:>5.2f}" for n in class_names)
        print(f"{r['threshold']:>5.2f}{r['coverage']:>10.3f}{acc:>9}  {per_class}")


def print_reliability(rows):
    print("\nTop score vs. actual accuracy")
    print(f"{'bin':<11}{'count':>7}{'mean score':>12}{'accuracy':>10}")
    for r in rows:
        if r["count"]:
            print(f"{r['low']:.2f}-{r['high']:.2f}  {r['count']:>7}{r['mean_confidence']:>12.3f}{r['accuracy']:>10.3f}")


def save_confusion_matrix(summary, title, path):
    cm = np.array(summary["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=summary["class_names"]).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format=".2f"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sweep_figure(rows, class_names, path):
    thresholds = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(thresholds, [r["coverage"] for r in rows], label="coverage (share labeled)")
    axes[0].plot(thresholds, [r["accuracy_on_accepted"] for r in rows], label="accuracy on labeled")
    axes[0].set(xlabel="threshold", title="Overall (validation)")
    axes[0].legend()
    for name in class_names:
        axes[1].plot(thresholds, [r["per_class_coverage"][name] for r in rows], label=name)
    axes[1].set(xlabel="threshold", ylabel="share of that class labeled", title="Coverage per true class")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--final", action="store_true", help="required for the test split")
    parser.add_argument("--reader", choices=["tf", "cv2"], default="tf", help="image decoder")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        raise SystemExit(
            "The test set is for the final, one-time evaluation.\n"
            "Freeze the model and threshold first, then run: python evaluate.py --split test --final"
        )

    predictor = EmotionPredictor(MODEL_PATH, LABELS_PATH)
    class_names = predictor.labels
    paths = validation_paths(class_names) if args.split == "val" else test_paths(class_names)

    print(f"Scoring {len(paths)} {args.split} images ({args.reader} decoder, the app's preprocessing)...")
    start = time.perf_counter()
    scores, kept, skipped = score_images(paths, predictor, reader=args.reader)
    print(f"Done in {time.perf_counter() - start:.1f} s; skipped {skipped} unreadable image(s).")

    y_true = np.array([class_names.index(p.parent.name) for p in kept])
    y_pred = scores.argmax(axis=1)

    summary = classification_summary(y_true, y_pred, class_names)
    print_report(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    save_confusion_matrix(
        summary, f"{args.split} confusion matrix (rows = true class)", RESULTS_DIR / f"{args.split}_confusion_matrix.png"
    )
    record = {
        "split": args.split,
        "images": len(kept),
        "skipped": skipped,
        "reader": args.reader,
        "model_file": MODEL_PATH,
        "keras_version": keras.__version__,
        "summary": summary,
    }
    (RESULTS_DIR / f"{args.split}_metrics.json").write_text(json.dumps(record, indent=2))

    if args.split == "val":
        sweep = threshold_sweep(scores, y_true, class_names, np.round(np.arange(0.0, 0.91, 0.05), 2))
        reliability = reliability_table(scores, y_true, [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        print_sweep(sweep, class_names)
        print_reliability(reliability)
        save_sweep_figure(sweep, class_names, RESULTS_DIR / "val_threshold_sweep.png")
        (RESULTS_DIR / "val_calibration.json").write_text(
            json.dumps({"threshold_sweep": sweep, "reliability": reliability}, indent=2)
        )

    print(f"\nSaved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
    
