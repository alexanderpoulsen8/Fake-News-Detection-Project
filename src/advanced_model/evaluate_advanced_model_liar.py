from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

LIAR_TEST_PATH = Path("data/liar/test.tsv")
MODEL_PATH = Path("data/models/advanced_model.joblib")
RESULTS_PATH = Path("results/advanced_model_liar_metrics.txt")
CONFUSION_PATH = Path("results/advanced_model_liar_confusion_matrix.csv")

# LIAR columns:
# 0: id
# 1: label
# 2: statement
# ... rest are metadata

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

FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}


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

    X_test = liar_df["statement"].astype(str)
    y_test = liar_df["binary_label"]

    print("Loading trained advanced model...")
    model = joblib.load(MODEL_PATH)

    print("Predicting on LIAR test set...")
    y_pred = model.predict(X_test)

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