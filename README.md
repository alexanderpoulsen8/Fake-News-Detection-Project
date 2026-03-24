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
│   ├── logistic_model.pkl
│   ├── models
│   ├── processed
│   ├── results
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
│   ├── descriptive_stats
│   ├── pipeline
│   ├── setup_nltk.py
│   └── subset_test.py
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share
```