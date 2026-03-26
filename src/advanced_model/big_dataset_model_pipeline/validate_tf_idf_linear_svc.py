import tf_idf_vectorizer as Vec
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import joblib

start_path = Path.cwd().parents[2]
data_dir = start_path / 'data' / 'big_dataset'
_VAL_PATH = data_dir / 'big_preprocessed_split' / 'val.csv'
_MODEL_PATH = data_dir / 'models' / 'tf_idf_linear_svc.joblib'
_OUTPUT_RESULTS_PATH = data_dir / 'results' / 'tf_idf_linear_svc_metrics.txt'



def main(
    val_path=_VAL_PATH,
    model_path=_MODEL_PATH,
    output_path=_OUTPUT_RESULTS_PATH
):
    X, y = Vec.vectorize_articles(val_path)
    clf = joblib.load(model_path)

    print('\nPredicting y values...')
    y_pred = clf.predict(X)

    print('\nComputing F1-score...')
    f1 = f1_score(y, y_pred, average='weighted')
    report = classification_report(y, y_pred)

    print(f"\nWeighted validation F1: {f1:.4f}")
    print(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Validation F1: {f1:.4f}\n\n")
        f.write(f"Val path: {val_path}\n")
        f.write("TF-IDF config:\n")
        f.write("  Absolute min_df=2500\n")
        f.write("  Relative max_df=0.8\n")
        f.write("  ngram_range=(1,2)\n")
        f.write("  sublinear_tf=True\n")
        f.write("  smooth_idf=True\n")
        f.write("LinearSVC config:\n")
        f.write(f"  C=0.5\n\n")
        f.write(report)
    print(f"Saved results to {output_path}")



if __name__ == '__main__':
    main()