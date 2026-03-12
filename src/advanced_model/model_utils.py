from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


FAKE_LABELS = {
    "fake", "conspiracy", "hate", "junksci", "unreliable", "clickbait", "bias", "satire"
}

REAL_LABELS = {
    "reliable", "political"
}


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def map_labels(df: pd.DataFrame, label_col: str = "type") -> pd.DataFrame:
    df = df.copy()

    def to_binary(label: str):
        if pd.isna(label):
            return None
        label = str(label).strip().lower()
        if label in FAKE_LABELS:
            return 1
        if label in REAL_LABELS:
            return 0
        return None

    df["label"] = df[label_col].apply(to_binary)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def build_advanced_pipeline(
    max_features: int = 50000,
    min_df: int = 5,
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    c: float = 1.0,
) -> Pipeline:
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                ngram_range=ngram_range,
                sublinear_tf=True,
                stop_words=stop_words,
            ),
        ),
        (
            "clf",
            LinearSVC(C=c, random_state=42),
        ),
    ])
    return pipeline


def save_model(model, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)