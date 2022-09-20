# # WIP - Restaurant Recommendation Dialog System
# ## Members:
# - Karpiński, R.R. (Rafał)
# - Pavan, L. (Lorenzo)
# - Rodrigues Luchetti, G.L. (Gustavo)
# - Teunissen, N.D. (Niels)

import re
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
    label_dict = df[['label', 'label_id']].drop_duplicates(
    ).set_index('label_id')

    return df, label_dict


# first baseline system - "always assigns the majority class of in the data"
majority_class = "inform"
def set_majority_class(df):
    print(df, majority_class)
    """Returns majority label for a given dataframe with a label column"""
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
    return (model, tfidf)


def train_logistic_regression_model(df):
    return train_model(LogisticRegression(random_state=10, max_iter=400), df)


def train_NB_classifier_model(df):
    return train_model(MultinomialNB(), df)


logistic_regression = None
def predict_lr(utterance):
    model = logistic_regression
    return model.predict(utterance)


multinomial_nb = None
def predict_nb(utterance):
    model = multinomial_nb
    return model.predict(utterance)


def main():
    """Prepares the dataset, model and runs the bot"""
    df = get_dataset('dialog_acts.dat')
    print(f"Dataset loaded into Dataframe! \n {df.head}")
    df_proc, label_dict = preprocess(df)
    df = df_proc.copy()
    print(f"\nDataframe after processing: \n {df.head}")

    logistic_regression, lr_vectorizer = train_logistic_regression_model(df)
    multinomial_nb, nb_vectorizer = train_NB_classifier_model(df)

    # all classifiers should support a single string as argument
    # and output a label as a classification prediction
    list_models = {
        "1": get_majority_class,
        "2": single_keyword_matching,
        "3": logistic_regression,
        "4": multinomial_nb
    }
    bot(list_models)


main()
