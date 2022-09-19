#!/usr/bin/env python
# coding: utf-8

# # WIP - Restaurant Recommendation Dialog System
# ## Members:
# - Karpiński, R.R. (Rafał)
# - Pavan, L. (Lorenzo)
# - Rodrigues Luchetti, G.L. (Gustavo)
# - Teunissen, N.D. (Niels)

# In[1]:

from bot import bot
import pandas as pd
import numpy as np
import re

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# In[2]:


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
# In[3]:


def preprocess(df):
    """
    Preprocesses the dataset
    args: DataFrame dataset
    return: dataset
    """
    df_proc = df.copy()
    print("before removing 'noise': ")
    print(df_proc.describe())

    # if text contains "noise" or "tv_noise", remove it
    for index, row in df_proc.iterrows():
        if "tv_noise" in row[1]:
            row[1] = row[1].replace('tv_noise', '')
        elif "noise" in row[1]:
            row[1] = row[1].replace('noise', '')
        row[1] = row[1].strip()

    # if that makes the column empty, remove the row
    for i in range(len(df_proc.index)):
        text = df['text'][i]
        if not text:
            df_proc = df.drop([i])

    print("\nafter: ")
    print(df_proc.describe())
    # making label dict (turning labels into numbers)
    df['label_id'] = df['label'].factorize()[0]
    label_dict = df[['label', 'label_id']].drop_duplicates(
    ).set_index('label_id')

    return df_proc, label_dict
# In[4]:

# should we remove null completely?
keyword_dict = {
    "inform": "\blooking for\b|\bdont care\b|\bdoesnt matter\b|\bexpensive\b|\bcheap\b|\bmoderate\b|\bi need\b|\bi want\b|\bfood\b|\bnorth\b",
    "confirm": "\bdoes it\b|\bis it\b|\bdo they\b|\bis that\b|\bis there\b",
    "affirm": "\byes\b|\byeah\b|\bcorrect\b",
    "request": "\bwhat is\b|\bwhats\b|\bmay i\b|\bcould i\b|\bwhat\b|\bprice range\b|\bpost code\b|\btype of\b|\baddress\b|\bphone number\b|\bcan i\b|\bcould i\b|\bcould you\b|\bdo you\b|\bi want+.address\b|\bi want+.phone\b|\bi would\b|\bwhere is\b",
    "thankyou": "\bthank you\b",
    "bye": "\bgoodbye\b|\bbye\b",
    "reqalts": "\bhow about\b|\bwhat about\b|\banything else\b|\bare there\b|\bis there\b|\bwhat else\b",
    "negate": "\bno\b|\bnot\b",
    "hello": "\bhello\b",
    "repeat": "\brepeat\b",
    "ack": "\bokay\b|\bkay\b",
    "restart": "\bstart\b",
    "deny": "\bdont\b",
    "reqmore": "\bmore\b",
    "null": "_?_",
}
# In[ ]:

# first baseline system - "always assigns the majority class of in the data"
def get_majority_class(df):
    """
    Classifies dialog based on the majority class label.
    args: utterance (any string)
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    majority_class = df['label'].mode().to_string(index=False)
    print(f"Majority class is '{majority_class}'")
    return majority_class


def df_majority_class(dataframe):
    """
    Classifies dialog based on the majority class label.
    args: pandas DataFrame that contains a column named text with utterances.
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    predictions = []
    for i in range(0, len(dataframe)):
        predictions.append(get_majority_class())
    return predictions
# In[6]:

# second baseline system - "rule-based system based on keyword matching"
def single_keyword_matching(text):
    """
    Rule-based prediction of a dialog act based on a phrase.
    args: utterance (any string)
    returns: Returns the predicted dialog act.
    """
    label = "inform"
    for key in keyword_dict:
        # if we find one of our keywords on any given string
        if (re.search(keyword_dict[key], text)):
            label = key
            return
    return label


def df_keyword_matching(df):
    """
    Rule-based prediction of dialog acts based on a colletion of utterances.
    args: DataFrame that contains a column named text with utterances.
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    predictions = []
    for i in range(0, len(df)):
        text = df.loc[i, 'text']
        predictions.append(single_keyword_matching(text))
    return predictions
# In[19]:


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

    X_test_tfidf = tfidf.transform(X_test).toarray()
    y_pred_test = model.predict(X_test_tfidf)

    return (model, tfidf)


def train_logistic_regression_model(df):
    return train_model(LogisticRegression(random_state=10, max_iter=400), df)


def train_NB_classifier_model(df):
    return train_model(MultinomialNB(), df)

# In[23]:


def main(classifier):
    """Prepares the dataset, model and runs the bot"""
    df = get_dataset('dialog_acts.dat')
    df, label_dict = preprocess(df)

    print('\nDF after processing: ')
    print(df.head(3))
    lr, lr_vectorizer = train_logistic_regression_model(df)
    nb, nb_vectorizer = train_NB_classifier_model(df)
    bot(lr, lr_vectorizer, label_dict)


main("test")
