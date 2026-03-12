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


TEST_PATH = Path("data/test.csv")
MODEL_PATH = Path("data/models/advanced_model.joblib")
RESULTS_PATH = Path("results/advanced_model_test_metrics.txt")
CONFUSION_PATH = Path("results/advanced_model_confusion_matrix.csv")


FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "clickbait", "unreliable", "bias", "satire", "political"
}

REAL_LABELS = {
    "reliable"
}


def map_label(label):
    if pd.isna(label):
        return None
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None


def prepare_df(df, text_col="content", label_col="type"):
    required_cols = [text_col, label_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    df["label"] = df[label_col].apply(map_label)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def main():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    text_col = "content"
    label_col = "type"

    test_df = prepare_df(test_df, text_col=text_col, label_col=label_col)

    X_test = test_df[text_col].astype(str)
    y_test = test_df["label"]

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(report)
    print("Confusion matrix:")
    print(cm)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("Advanced model test-set evaluation\n\n")
        f.write(f"Model path: {MODEL_PATH}\n")
        f.write(f"Test path: {TEST_PATH}\n\n")
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