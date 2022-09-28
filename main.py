# #Restaurant Recommendation Dialog System
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
import pickle

from bot import bot  # local import

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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


def build_label_dict(df):
    global label_dict
    # making label dict (turning labels into numbers)
    df['label_id'] = df['label'].factorize()[0]
    label_dict = df[['label_id', 'label']].drop_duplicates().set_index('label_id')
    return label_dict


def preprocess(df):
    """
    Preprocesses the dataset
    args: DataFrame dataset
    return: dataset
    """
    global label_dict
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
    return df


# first baseline system - "always assigns the majority class of in the data"
def set_majority_class(df):
    global majority_class
    """Returns majority label for a given dataframe with a label column"""
    print(df, majority_class)
    majority_class = df['label'].mode().to_string()
    return majority_class


def get_majority_class(utterance):
    global majority_class
    return majority_class


# second baseline system - "rule-based system based on keyword matching"
def single_keyword_matching(utterance):
    """
    Rule-based prediction of a dialog act based on a phrase.
    args: utterance (any string)
    returns: Returns the predicted dialog act.
    """
    label = "null"
    global keyword_dict
    for key in keyword_dict:
        # if we find one of our keywords on any given string
        if re.search(keyword_dict[key], utterance):
            label = key
            return label


def make_train_test_split(df):
    global tfidf
    # X - independent features (excluding target variable).
    # y - dependent variables (target we're looking to predict).
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_id'], test_size=0.15, random_state=42
    )

    # train test split includes vectorizing
    tfidf = TfidfVectorizer(
        sublinear_tf=True,  # scale the words frequency in logarithmic scale
        min_df=5,  # remove the words which has occurred in less than ‘min_df’ number of files
        ngram_range=(1, 2),  # don't know what role n-grams play in vectorisation
        stop_words='english',  # it removes stop words which are predefined in ‘english’.
        lowercase=True  # everything to lowercase
    )
    return tfidf.fit_transform(X_train).toarray(), X_test, y_train, y_test


# model specific functions
def train_logistic_regression_model(X_train, y_train):
    return LogisticRegression(max_iter=400).fit(X_train, y_train)


def train_NB_classifier_model(X_train, y_train):
    return MultinomialNB().fit(X_train, y_train)


# this is a bit convoluted, but this is the easiest way for us to retrieve
# a prediction's label given one of the models' predictions
# TODO: unify prediction functions into one?
def tfidf_convert(utterance):
    global tfidf
    array = np.array([utterance])
    converted = tfidf.transform(array).toarray()
    return converted


def return_pred(label_id):
    pred = label_dict['label'].iloc[label_id.item(0)]
    print(f"Label ID: {label_id} \nActual Prediction: {pred}")
    return pred


def predict_lr(utterance):
    global logistic_regression, tfidf, label_dict
    tfidf_ut = tfidf_convert(utterance)
    return return_pred(
        logistic_regression.predict(tfidf_ut)
    )


def predict_mnb(utterance):
    global multinomial_nb, tfidf, label_dict
    tfidf_ut = tfidf_convert(utterance)
    return return_pred(
        multinomial_nb.predict(tfidf_ut)
    )


def main():
    """Prepares the dataset, model and runs the bot"""
    global label_dict, tfidf
    global logistic_regression, multinomial_nb
    models_path = 'models/'
    source_data = 'dialog_acts.dat'
    df_file = 'df.csv'

    print(f"Arguments: {sys.argv}")
    if os.path.exists(df_file) and "reprocess" not in sys.argv:
        print("Found existing processed dataframe! Using it instead...")
        df = pd.read_csv(df_file)
    else:
        print("Building and processing dataframe from scratch.")
        df = get_dataset(source_data)
        print(f"Dataset loaded into Dataframe! \n {df.describe()}")

        df_proc = preprocess(df)
        df = df_proc.copy()
        print(f"\nDataframe after processing: \n {df.describe()}")

        df.to_csv(df_file, index=False)
        print("Processed dataframe saved as .csv! \n")

    label_dict = build_label_dict(df)
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    if "remodel" not in sys.argv:
        if os.path.exists(models_path+"lr.sav"):
            print("Reusing Logistic Regression model...")
            logistic_regression = pickle.load(open(models_path+'lr.sav', 'rb'))
        if os.path.exists(models_path+"mnb.sav"):
            print("Reusing Multinomial NB model...")
            multinomial_nb = pickle.load(open(models_path+'mnb.sav', 'rb'))
    else:
        print("Building models from scratch! This might take a while.")
        logistic_regression = train_logistic_regression_model(X_train, y_train)
        multinomial_nb = train_NB_classifier_model(X_train, y_train)

        print("Models have been fit! Saving them for future use... \n")
        pickle.dump(logistic_regression, open(models_path+'lr.sav', 'wb'))
        pickle.dump(multinomial_nb, open(models_path+'mnb.sav', 'wb'))

    # the following functions have a single string as their argument
    # and return a label as a classification prediction
    list_models = {
        "1": get_majority_class,
        "2": single_keyword_matching,
        "3": predict_lr,
        "4": predict_mnb
    }
    bot(list_models)


global majority_class, keyword_dict
majority_class = "inform"
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

main()
