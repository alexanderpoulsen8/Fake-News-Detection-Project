# Advanced Model Pipeline - Modular Architecture

## Overview

The advanced model pipeline uses TF-IDF vectorization with LinearSVC for fake news detection. The pipeline has been refactored into a clean, modular architecture with separate modules for each step.

## Project Structure

```
src/
├── Advanced_Main.py            # Main orchestrator (clean and simple)
├── pipeline/
│   ├── preprocessor.py         # Step 1: Preprocess raw CSV (optional)
│   ├── data_splitter.py        # Step 2: Split data into train/val/test
│   ├── advanced_trainer.py     # Step 3: Train TF-IDF + LinearSVC model
│   └── advanced_evaluator.py   # Step 4: Evaluate on all splits
└── advanced_model/             # Legacy scripts (kept for reference)
```

## Usage

### Quick Start

1. **Edit configuration** in `Advanced_Main.py`:
```python
DATA_DIR = r"D:\GDS\Fake-News-Detection-Project\data"
RESULTS_DIR = r"D:\GDS\Fake-News-Detection-Project\results"
MODEL_NAME = "advanced_model"

# Optional preprocessing
RAW_CSV_PATH = None  # Set to raw CSV path if needed

# Hyperparameters
SAMPLE_SIZE = None  # None = use all data, or set to integer
MAX_FEATURES = 30000
MIN_DF = 5
NGRAM_RANGE = (1, 2)
C_VALUE = 0.5
```

2. **Run the pipeline**:
```bash
cd d:\GDS\Fake-News-Detection-Project\src
python Advanced_Main.py
```

### What It Does

The pipeline automatically:
1. ✅ (Optional) Preprocesses raw CSV with text cleaning and tokenization
2. ✅ Splits data into train/val/test (80/10/10) - **skips if already done**
3. ✅ Trains TF-IDF + LinearSVC model
4. ✅ Evaluates on train/val/test splits with overfitting diagnosis
5. ✅ Saves results to `results/` folder

### Output Files

**Data folder** (`data/`):
- `preprocessed_dataset.csv` - Preprocessed text (if preprocessing was run)
- `train.csv`, `val.csv`, `test.csv` - Data splits

**Results folder** (`results/`):
- `{model_name}.joblib` - Trained model
- `{model_name}_training_results.txt` - Training metrics
- `{model_name}_full_evaluation.txt` - Complete evaluation report
- `{model_name}_evaluation.csv` - Metrics in CSV format

## Pipeline Steps

### Step 1: Preprocessing (Optional)
**Module:** `preprocessor.py`  
**Function:** `preprocess_dataset(input_file, output_file, text_column)`

- Cleans text (URLs, emails, dates, numbers → tokens)
- Tokenizes with NLTK
- Removes stopwords (keeps no/nor/not)
- Stems with Snowball stemmer
- Auto-skips if preprocessed file exists

**When to use:**
- Set `RAW_CSV_PATH` if you have a raw CSV that needs preprocessing
- Set to `None` if you already have `preprocessed_dataset.csv`

### Step 2: Data Splitting
**Module:** `data_splitter.py`  
**Function:** `split_data(data_dir)`

- Same as simple model pipeline
- Deterministic hash-based splitting (80/10/10)
- Auto-skips if splits exist

### Step 3: Training
**Module:** `advanced_trainer.py`  
**Function:** `train_advanced_model(data_dir, results_dir, model_name, ...)`

**Model Architecture:**
- **TF-IDF Vectorizer:**
  - Converts text to TF-IDF features
  - Configurable max_features, min_df, ngram_range
  - Uses sublinear TF scaling
  - Removes English stopwords
  
- **LinearSVC:**
  - Linear Support Vector Classifier
  - Fast and effective for text classification
  - Configurable C parameter for regularization

**Hyperparameters:**
```python
MAX_FEATURES = 30000    # Max TF-IDF features
MIN_DF = 5              # Min document frequency
NGRAM_RANGE = (1, 2)    # Unigrams + bigrams
C_VALUE = 0.5           # SVM regularization
SAMPLE_SIZE = None      # Training sample size (None = all)
```

**Returns:**
```python
{
    'val_f1': 0.9234,
    'model_path': 'results/advanced_model.joblib',
    'results_path': 'results/advanced_model_training_results.txt'
}
```

### Step 4: Evaluation
**Module:** `advanced_evaluator.py`  
**Function:** `evaluate_advanced_model(data_dir, results_dir, model_name)`

- Evaluates on train/val/test splits
- Calculates F1, precision, recall, accuracy
- Generates confusion matrices
- Diagnoses overfitting/underfitting
- Saves detailed reports

**Returns:**
```python
{
    'train_f1': 0.9456,
    'val_f1': 0.9234,
    'test_f1': 0.9198,
    'test_accuracy': 0.9201,
    'diagnosis': ['No strong sign of overfitting...']
}
```

## Running Different Experiments

### Quick training with sample:
```python
SAMPLE_SIZE = 100000  # Use 100K samples for faster training
MODEL_NAME = "advanced_model_quick"
```

### Full training:
```python
SAMPLE_SIZE = None  # Use all data
MODEL_NAME = "advanced_model_full"
```

### Different hyperparameters:
```python
# Experiment 1: More features
MAX_FEATURES = 50000
MODEL_NAME = "advanced_model_50k"

# Experiment 2: Trigrams
NGRAM_RANGE = (1, 3)
MODEL_NAME = "advanced_model_trigrams"

# Experiment 3: Stronger regularization
C_VALUE = 0.1
MODEL_NAME = "advanced_model_c01"
```

## Model Comparison

| Feature | Simple Model | Advanced Model |
|---------|-------------|----------------|
| Algorithm | Logistic Regression | LinearSVC |
| Features | Bag-of-Words counts | TF-IDF |
| N-grams | Unigrams only | Unigrams + Bigrams |
| Stopwords | Manual removal | Built-in removal |
| Typical F1 | ~0.86 | ~0.92 |
| Training Time | Faster | Slower |
| Memory | Lower | Higher |

## Troubleshooting

**Error: "Preprocessed dataset not found"**
- Either set `RAW_CSV_PATH` to preprocess raw data
- Or ensure `preprocessed_dataset.csv` exists in data folder

**Memory errors during training**
- Reduce `MAX_FEATURES` (e.g., 20000 or 10000)
- Use `SAMPLE_SIZE` to train on subset
- Close other applications

**Training too slow**
- Use `SAMPLE_SIZE` for faster experiments
- Reduce `MAX_FEATURES`
- Consider using simple model for quick iterations

**Poor performance**
- Try different `C_VALUE` (0.1, 0.5, 1.0, 2.0)
- Increase `MAX_FEATURES`
- Add trigrams: `NGRAM_RANGE = (1, 3)`
- Check class balance in training data

## Example Workflow

```python
# 1. Quick experiment with sample
SAMPLE_SIZE = 50000
MAX_FEATURES = 10000
MODEL_NAME = "quick_test"
# Run → Check results

# 2. Full training with best hyperparameters
SAMPLE_SIZE = None
MAX_FEATURES = 30000
C_VALUE = 0.5
MODEL_NAME = "advanced_model_final"
# Run → Evaluate

# 3. Compare results
# Check results/advanced_model_final_evaluation.csv
```

## Advanced Usage

You can also import and use modules directly:

```python
from pipeline.advanced_trainer import train_advanced_model
from pipeline.advanced_evaluator import evaluate_advanced_model

# Custom training
results = train_advanced_model(
    data_dir="path/to/data",
    results_dir="path/to/results",
    model_name="custom_model",
    max_features=50000,
    ngram_range=(1, 3),
    c_value=1.0
)

# Evaluate
eval_results = evaluate_advanced_model(
    data_dir="path/to/data",
    results_dir="path/to/results",
    model_name="custom_model"
)

print(f"Test F1: {eval_results['test_f1']:.4f}")
```

## Performance Tips

1. **Start with a sample** - Use `SAMPLE_SIZE=100000` for quick iterations
2. **Tune hyperparameters** - Try different C values and feature counts
3. **Monitor overfitting** - Check the fit diagnosis in evaluation results
4. **Compare models** - Train multiple versions and compare CSV results
5. **Use full data for final model** - Set `SAMPLE_SIZE=None` for production
