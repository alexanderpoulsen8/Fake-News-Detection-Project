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
├── Notebooks_CSV_analysis
│   ├── bigdata
│   └── smalldata
├── README.md
├── data
│   ├── liar
│   ├── models
│   ├── processed
│   ├── results
│   ├── temp.txt
│   ├── test.csv
│   ├── tmp.csv
│   ├── top_10000_vocab.pkl
│   ├── train.csv
│   ├── val.csv
│   ├── vocab.pkl
│   └── vocabulary.csv
├── docs
│   ├── p1_data_processing.ipynb
│   ├── preprocessing.py
│   └── print_top_100_words.py
├── requirements.txt
├── results
│   ├── $advanced_model_liar_confusion_matrix.csv
│   ├── advanced_model_liar_confusion_matrix.csv
│   └── advanced_model_liar_metrics.txt
├── src
│   ├── Simple_Main.py
│   ├── advanced_model
│   ├── descriptive_stats
│   ├── pipeline
│   ├── setup_nltk.py
│   └── validate_simple_model.py
└── venv
    ├── bin
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share
```
