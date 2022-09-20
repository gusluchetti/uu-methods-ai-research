# # WIP - Restaurant Recommendation Dialog System
# ## Members:
# - Karpiński, R.R. (Rafał)
# - Pavan, L. (Lorenzo)
# - Rodrigues Luchetti, G.L. (Gustavo)
# - Teunissen, N.D. (Niels)

import re
import os
import sys
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from bot import bot  # local import

# main should be responsible for setting up the dataset, doing preprocessing
# and training the models that will be used
# those models are then passed onto the bot that is called at the very end
# of the file, with these aformentioned baseline systems + models


def get_dataset(path):
    """
    Getting dataset from .dat file
    args: dataset file path
    return: DataFrame of dataset
    """
    with open(path, 'r') as f:
        data = f.readlines()
        data = list(map(lambda x: x.rstrip("\n").split(" ", 1), data))

    df = pd.DataFrame(np.array(data), columns=['label', 'text'])
    return df


def preprocess(df):
    global label_dict
    """
    Preprocesses the dataset
    args: DataFrame dataset
    return: dataset
    """
    # if text contains "noise" or "tv_noise", remove it
    for index, row in df.iterrows():
        if "tv_noise" in row[1]:
            row[1] = row[1].replace('tv_noise', '')
        elif "noise" in row[1]:
            row[1] = row[1].replace('noise', '')
        row[1] = row[1].strip()

    # if that makes the column empty, remove the row
    for i in range(len(df.index)):
        text = df['text'][i]
        if not text:
            df = df.drop([i])

    # making label dict (turning labels into numbers)
    df['label_id'] = df['label'].factorize()[0]
    label_dict = df[['label', 'label_id']].drop_duplicates().set_index('label_id')
    print(f"\nLabel Dictionary = {label_dict}")
    return df, label_dict


# first baseline system - "always assigns the majority class of in the data"
majority_class = "inform"
def set_majority_class(df):
    """Returns majority label for a given dataframe with a label column"""
    print(df, majority_class)
    return df['label'].mode().to_string()


def get_majority_class(utterance):
    return majority_class

# should we remove null completely?
keyword_dict = {
    "inform": r"\blooking\b|\blooking for\b|\bdont care\b|\bdoesnt matter\b|\bexpensive\b|\bcheap\b|\bmoderate\b|\bi need\b|\bi want\b|\bfood\b|\bnorth\b",
    "confirm": r"\bdoes it\b|\bis it\b|\bdo they\b|\bis that\b|\bis there\b",
    "affirm": r"\byes\b|\byeah\b|\bcorrect\b",
    "request": r"\bwhat is\b|\bwhats\b|\bmay i\b|\bcould i\b|\bwhat\b|\bprice range\b|\bpost code\b|\btype of\b|\baddress\b|\bphone number\b|\bcan i\b|\bcould i\b|\bcould you\b|\bdo you\b|\bi want+.address\b|\bi want+.phone\b|\bi would\b|\bwhere is\b",
    "thankyou": r"\bthank you\b",
    "bye": r"\bgoodbye\b|\bbye\b",
    "reqalts": r"\bhow about\b|\bwhat about\b|\banything else\b|\bare there\b|\bis there\b|\bwhat else\b",
    "negate": r"\bno\b|\bnot\b",
    "hello": r"\bhello\b",
    "repeat": r"\brepeat\b",
    "ack": r"\bokay\b|\bkay\b",
    "restart": r"\bstart\b",
    "deny": r"\bdont\b",
    "reqmore": r"\bmore\b",
    "null": r"__?__",
}


# second baseline system - "rule-based system based on keyword matching"
def single_keyword_matching(utterance):
    """
    Rule-based prediction of a dialog act based on a phrase.
    args: utterance (any string)
    returns: Returns the predicted dialog act.
    """
    label = "null"
    for key in keyword_dict:
        # if we find one of our keywords on any given string
        if re.search(keyword_dict[key], utterance):
            label = key
            return label


def train_model(method, df):
    global tfidf
    """
    Trains any method of classifier that fits the pre-processing.
    args: model being used (and any parameters for said model), and a dataframe
    returns: tuple of a trained, fitted model and a NLP vectorizer/transformer
    """
    # X - independent features (excluding target variable).
    # y - dependent variables (target we're looking to predict).
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_id'], test_size=0.15, random_state=10
    )

    tfidf = TfidfVectorizer(
        sublinear_tf=True,  # scale the words frequency in logarithmic scale
        min_df=5,  # remove the words which has occurred in less than ‘min_df’ number of files
        ngram_range=(1, 2),  # don't know what role n-grams play in vectorisation
        stop_words='english',  # it removes stop words which are predefined in ‘english’.
        lowercase=True  # everything to lowercase
    )

    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    model = method
    model.fit(X_train_tfidf, y_train)

    # X_test_tfidf = tfidf.transform(X_test).toarray()
    # y_pred_test = model.predict(X_test_tfidf)
    return model


def train_logistic_regression_model(df):
    return train_model(LogisticRegression(random_state=10, max_iter=400), df)


def train_NB_classifier_model(df):
    return train_model(MultinomialNB(), df)


def tfidf_convert(utterance):
    array = np.array([utterance])
    converted = tfidf.transform(array).toarray()
    return converted

# this is a bit convoluted, but this is the easiest way for us to retrieve
# a prediction's label given one of the models' predictions
# TODO: unify prediction functions into one?
def predict_lr(utterance):
    global logistic_regression, tfidf, label_dict
    return label_dict[logistic_regression.predict(tfidf_convert(utterance)).item(0)]


def predict_mnb(utterance):
    global multinomial_nb, tifdf, label_dict
    return label_dict[multinomial_nb.predict(tfidf_convert(utterance)).item(0)]


def main():
    global tfidf
    global logistic_regression
    global multinomial_nb
    global saved_lr
    global saved_mnb

    """Prepares the dataset, model and runs the bot"""
    print(sys.argv)
    if os.path.exists("df.csv") and "reprocess" not in sys.argv:
        print("Found existing processed dataframe! Using it instead...")
        df = pd.read_csv("df.csv")
    else:
        print("Building and processing dataframe from scratch.")
        df = get_dataset('dialog_acts.dat')
        print(f"Dataset loaded into Dataframe! \n {df.head}")

        df_proc, label_dict = preprocess(df)
        df = df_proc.copy()
        print(f"\nDataframe after processing: \n {df.head}")

        df.to_csv("df.csv", index=False)
        print("Processed dataframe saved as .csv! \n")

    if "remodel" not in sys.argv:
        print("Found saved models! Reusing them...")
        logistic_regression = pickle.loads(saved_lr)
        multinomial_nb = pickle.loads(saved_mnb)
    else:
        print("Building models from scratch! This will take a while.")
        logistic_regression = train_logistic_regression_model(df)
        multinomial_nb = train_NB_classifier_model(df)
        saved_lr = pickle.dumps(logistic_regression)
        saved_mnb = pickle.dumps(multinomial_nb)

    # all classifiers should support a single string as argument
    # and output a label as a classification prediction
    list_models = {
        "1": get_majority_class,
        "2": single_keyword_matching,
        "3": predict_lr,
        "4": predict_mnb
    }
    bot(list_models)


main()
