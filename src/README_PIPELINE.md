# Simple Model Pipeline - Modular Architecture

## Overview

The pipeline has been refactored into a clean, modular architecture with separate modules for each step and a simple orchestrator script.

## Project Structure

```
src/
├── Simple_Main.py              # Main orchestrator (clean and simple)
├── pipeline/
│   ├── data_splitter.py        # Step 1: Split data into train/val/test
│   ├── vocab_builder.py        # Step 2: Build vocabulary from training data
│   └── model_trainer.py        # Step 3: Train model and save results
└── Simple_model.py             # Legacy standalone script (kept for reference)
```

## Usage

### Quick Start

1. **Edit configuration** in `Simple_Main.py`:
```python
DATA_DIR = r"D:\GDS\Fake-News-Detection-Project\data"
RESULTS_DIR = r"D:\GDS\Fake-News-Detection-Project\results"
TOP_K_WORDS = 10000
MODEL_NAME = "logistic_model"
```

2. **Run the pipeline**:
```bash
cd d:\GDS\Fake-News-Detection-Project\src
python Simple_Main.py
```

### What It Does

The pipeline automatically:
1. ✅ Checks if `preprocessed_dataset.csv` exists
2. ✅ Splits data into train/val/test (80/10/10) - **skips if already done**
3. ✅ Builds vocabulary of top K words - **skips if already done**
4. ✅ Trains logistic regression model
5. ✅ Saves results to `results/` folder

### Output Files

**Data folder** (`data/`):
- `train.csv`, `val.csv`, `test.csv` - Data splits
- `top_{K}_vocab.pkl` - Vocabulary

**Results folder** (`results/`):
- `{model_name}.pkl` - Trained model
- `{model_name}_vocab.pkl` - Vocabulary (copy for deployment)
- `{model_name}_results.txt` - Detailed results report
- `{model_name}_results.csv` - Results in CSV format

## Running Different Experiments

### Different vocabulary sizes:
```python
# In Simple_Main.py, change:
TOP_K_WORDS = 20000
MODEL_NAME = "model_20k"
```

### Multiple experiments:
```python
# Run 1: 10K words
TOP_K_WORDS = 10000
MODEL_NAME = "model_10k"

# Run 2: 20K words
TOP_K_WORDS = 20000
MODEL_NAME = "model_20k"

# Run 3: 30K words
TOP_K_WORDS = 30000
MODEL_NAME = "model_30k"
```

Each run will create separate result files in the `results/` folder.

## Pipeline Modules

### 1. `data_splitter.py`
**Function:** `split_data(data_dir, preprocessed_file, salt)`
- Splits preprocessed data using DuckDB
- Deterministic hash-based splitting
- Auto-skips if splits exist

### 2. `vocab_builder.py`
**Function:** `build_vocabulary(data_dir, top_k)`
- Counts word frequencies from training data
- Selects top K most frequent words
- Saves as pickle file
- Auto-skips if vocabulary exists

### 3. `model_trainer.py`
**Function:** `train_model(data_dir, results_dir, top_k, model_name)`
- Loads data and vocabulary
- Creates sparse feature matrices
- Trains logistic regression
- Evaluates on validation and test sets
- Saves model and results to results folder

**Returns:** Dictionary with performance metrics
```python
{
    'val_f1': 0.8622,
    'val_acc': 0.8600,
    'test_f1': 0.8619,
    'test_acc': 0.8598
}
```

## Advantages of This Architecture

✅ **Clean separation of concerns** - Each module does one thing
✅ **Easy to modify** - Change one module without affecting others
✅ **Reusable** - Import modules in other scripts
✅ **Simple orchestrator** - Main script is ~90 lines, easy to understand
✅ **Organized outputs** - Models and results in dedicated folder
✅ **Smart caching** - Skips expensive steps if already done

## Example: Custom Pipeline

You can also import and use the modules directly:

```python
from pipeline.data_splitter import split_data
from pipeline.vocab_builder import build_vocabulary
from pipeline.model_trainer import train_model

# Custom workflow
data_dir = "path/to/data"
results_dir = "path/to/results"

# Only build vocabulary for 5000 words
build_vocabulary(data_dir, top_k=5000)

# Train model
results = train_model(data_dir, results_dir, top_k=5000, model_name="small_model")
print(f"Test F1: {results['test_f1']:.4f}")
```

## Troubleshooting

**Error: "Preprocessed dataset not found"**
- Make sure you've run the preprocessing pipeline first
- Check that `preprocessed_dataset.csv` exists in the data folder

**Error: "Training data not found"**
- The data hasn't been split yet
- Let the pipeline run Step 1 (data splitting)

**Memory errors**
- The pipeline uses sparse matrices to minimize memory usage
- Should work with 16GB+ RAM
- For very large vocabularies (>30K), you may need 32GB RAM
