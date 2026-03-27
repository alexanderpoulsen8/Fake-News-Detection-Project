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
   - Make sure to have the 995k subset, the full Fake News Corpus dataset, and the LIAR set all in the same directory on your machine. All of the executive scripts throughout the repository uniquely specify directory paths of input data, output data and trained model files (```.joblib``` files). Make sure to specify these directory paths properly when running the scripts.

3. Run the model:
   - To run the models presented in this project, please see the ```src/``` directory. Scripts for training and evaluating both our logistic regression model and our advanved SGDClassifier are found here. Make sure to specify proper directory paths. Simply specify which dataset you wish to train, test and validate your data on and run the code.

4. View the results:
   - The script will output the results to the console in the form of ```scikit-learn```'s ```Classification report``` and save them in the `results/` directory.

```
├── Grouping.txt
├── README.md
├── TMP scripts
│   ├── CheckRows.py
│   └── PklToCsv.py
├── data
│   ├── big_dataset
│   │    ├── train.csv
│   │    ├── val.csv
│   │    ├── test.csv
│   │    └── tf_idf
│   │         ├── big_doc_freq_vector.csv
│   │         ├── big_pruned_doc_freq_vector.csv
│   │         ├── big_strict_pruned_doc_freq_vector.csv
│   │         └── idf_vector.csv
│   ├── LIAR
│   │    ├── train.tsv
│   │    ├── valid.tsv
│   │    ├── test.tsv
│   │    └── LIAR_dataset_combined.tsv
│   ├── small_dataset
│   │    ├── train.csv
│   │    ├── val.csv
│   │    └── test.csv
│   ├── models
│   │    ├── logistic_model.pkl
│   │    ├── SGDClassifier.joblib
│   │    └── LinearSVC.joblib
│   ├── results
│   ├── top_10000_vocab.pkl
│   ├── vocab.pkl
│   └── vocabulary.csv
├── docs
│   ├── p1_data_processing.ipynb
│   ├── preprocessing.py
|   └── print_top_100_words.py
├── requirements.txt
├── requirements_new.txt
├── src
│   ├── Advanced_Main.py
│   ├── Simple_Main.py
│   ├── setup_nltk.py
│   ├── pipeline
│   │    ├── data_splitter.py
│   │    ├── model_trainer.py
│   │    ├── news_sample.csv
│   │    ├── preprocess_with_duckdb.py
│   │    ├── preprocessing.py
│   │    └── vocab_builder.py
│   └── advanced_model
│        ├── __init__.py
│        ├── evaluate_advanced_model.py
│        ├── train_advanced_model.py
│        ├── train_advanced_model_chunked.py
│        └── big_dataset_model_pipeline
│             ├── build_doc_freq_vector.py
│             ├── df_to_idf.py
│             ├── LIAR_validate_tf_idf_SGDClassifier.py
│             ├── prune_ngrams_from_df.py
│             ├── strict_prun_ngrams_from_df.py
│             ├── tf_idf_vectorizer.py
│             ├── train_SGDClassifier.py
│             └── validate_tf_idf_SGDClassifier.py
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share
```
