# Fake News Detection Project
> During the GDS course, we have deveoloped a machine learning model, which is designed to classify news articles as reliable or fake based the articles' metadata.

## Running the model

To run the model, follow these steps:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   (or use our virtual environment).
2. Prepare your dataset:
   - Make sure to have the 995k subset (and the full dataset if you wish to run it) in the same directory on your machine. On line xxx in ```main.py```, you can specify the path to the directory which contains the data. Please note that the file names must match the expected format.

3. Run the model:
   - From the root directory of the project, run
   ```
   python3 main.py
   ```

4. View the results:
   - The script will output the results to the console and save them in the `results/` directory.
   
```
├── Grouping.txt
├── README.md
├── TMP scripts
│   ├── CheckRows.py
│   └── PklToCsv.py
├── data
│   ├── liar
│   │   ├── README
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── valid.tsv
│   ├── logistic_model.pkl
│   ├── models
│   │   ├── advanced_model.joblib
│   │   └── advanced_model_chunked.joblib
│   ├── processed
│   │   └── preprocessed_dataset.csv
│   ├── results
│   │   ├── advanced_model_chunked_metrics.txt
│   │   ├── advanced_model_full_evaluation.txt
│   │   └── advanced_model_metrics.txt
│   ├── test.csv
│   ├── tmp.csv
│   ├── top_10000_vocab.pkl
│   ├── train.csv
│   ├── val.csv
│   ├── vocab.pkl
│   └── vocabulary.csv
├── docs
│   └── temp.txt
├── inspo
│   ├── LongAss_logistic_model.pkl
│   └── assignment1.ipynb
├── requirements.txt
├── requirements_new.txt
├── scripts
│   ├── DataSplit.py
│   ├── build_vocab_from_stats.py
│   ├── setup_nltk.py
│   └── verify_vocab.py
├── src
│   ├── Advanced_Main.py
│   ├── Simple_Main.py
│   ├── Simple_model.py
│   ├── Simple_model_parallel.py
│   ├── advanced_model
│   │   ├── __init__.py
│   │   ├── evaluate_advanced_model.py
│   │   ├── evaluate_advanced_model_liar.py
│   │   ├── model_utils.py
│   │   ├── train_advanced_model.py
│   │   └── train_advanced_model_chunked.py
│   ├── descriptive_stats
│   │   ├── build_vocab.py
│   │   ├── reduce_vocab_stats.py
│   │   └── vocab_stats.py
│   ├── pipeline
│   │   ├── __pycache__
│   │   │   └── preprocessing.cpython-313.pyc
│   │   ├── data_splitter.py
│   │   ├── model_trainer.py
│   │   ├── news_sample.csv
│   │   ├── preprocess_with_duckdb.py
│   │   ├── preprocessing.py
│   │   ├── preprocessing_test.py
│   │   ├── preprocessor.py
│   │   ├── read_big_fckn_csv.py
│   │   └── vocab_builder.py
│   ├── setup_nltk.py
│   └── subset_test.py
└── venv
    ├── bin
    │   ├── Activate.ps1
    │   ├── activate
    │   ├── activate.csh
    │   ├── activate.fish
    │   ├── f2py
    │   ├── fonttools
    │   ├── nltk
    │   ├── numpy-config
    │   ├── pip
    │   ├── pip3
    │   ├── pip3.13
    │   ├── pyftmerge
    │   ├── pyftsubset
    │   ├── python -> python3
    │   ├── python3 -> /opt/miniconda3/bin/python3
    │   ├── python3.13 -> python3
    │   ├── tqdm
    │   ├── ttx
    │   └── wsdump
    ├── include
    │   └── python3.13
    ├── lib
    │   └── python3.13
    │       └── site-packages
    ├── pyvenv.cfg
    └── share
        └── man
            └── man1
```