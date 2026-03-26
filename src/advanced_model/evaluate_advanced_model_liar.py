from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix

# Ensure project root is on sys.path so `from src...` imports work when running
# this script directly (python src/advanced_model/evaluate_advanced_model_liar.py).
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_model.big_dataset_model_pipeline import tf_idf_vectorizer as Vec

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

LIAR_TEST_PATH = Path("data/liar/liar_dataset/liar_dataset_combined.tsv")
MODEL_PATH = Path("data/models/SGDClassifier.joblib")
RESULTS_PATH = Path("results/advanced_model_liar_metrics.txt")
CONFUSION_PATH = Path("results/advanced_model_liar_confusion_matrix.csv")

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

FAKE_LABELS = {"pants-fire", "false", "barely-true","half-true", "mostly-true"}
REAL_LABELS = {"true"}


def map_liar_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None


def prepare_liar_df(df):
    required_cols = ["statement", "label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["statement", "label"]).copy()
    df["binary_label"] = df["label"].apply(map_liar_label)
    df = df.dropna(subset=["binary_label"]).copy()
    df["binary_label"] = df["binary_label"].astype(int)
    return df


def vectorize_texts(texts):
    rows, cols, data = [], [], []

    for row_idx, text in enumerate(texts):
        tokens = str(text).split(" ")
        indices, values = Vec.vectorize_doc(tokens, ngram_range=(1, 2), sublinear=True)

        rows.extend([row_idx] * len(indices))
        cols.extend(indices)
        data.extend(values)

    X = csr_matrix((data, (rows, cols)), shape=(len(texts), len(Vec.vocab_idx)))
    return X


def main():
    print("Loading LIAR test data...")
    liar_df = pd.read_csv(
        LIAR_TEST_PATH,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        low_memory=False
    )

    liar_df = prepare_liar_df(liar_df)

    X_test_text = liar_df["statement"].astype(str).tolist()
    y_test = liar_df["binary_label"].to_numpy()

    print("Vectorizing LIAR statements with custom TF-IDF pipeline...")
    X_test_vec = vectorize_texts(X_test_text)

    print("Loading trained SGDClassifier...")
    model = joblib.load(MODEL_PATH)

    print("Predicting on LIAR test set...")
    y_pred = model.predict(X_test_vec)

    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"LIAR F1: {test_f1:.4f}")
    print(f"LIAR Precision: {test_precision:.4f}")
    print(f"LIAR Recall: {test_recall:.4f}")
    print(f"LIAR Accuracy: {test_accuracy:.4f}")
    print(report)
    print("Confusion matrix:")
    print(cm)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Advanced model LIAR evaluation\n\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"LIAR test path: {LIAR_TEST_PATH}\n\n")
        f.write("LIAR label mapping:\n")
        f.write("  fake: pants-fire, false, barely-true\n")
        f.write("  reliable: half-true, mostly-true, true\n\n")
        f.write(f"F1: {test_f1:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)

    cm_df = pd.DataFrame(
        cm,
        index=["true_reliable", "true_fake"],
        columns=["pred_reliable", "pred_fake"]
    )
    cm_df.to_csv(CONFUSION_PATH, index=True)

    print(f"Saved metrics to {RESULTS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_PATH}")


if __name__ == "__main__":
    main()