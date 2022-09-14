import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def get_dataset(path):
  """
  arg: dataset file path
  
  return: DataFrame of dataset
  """
  with open(path, 'r') as f:
    data = f.readlines()
    data = list(map(lambda x: x.rstrip("\n").split(" ", 1), data))
  df = pd.DataFrame(np.array(data), columns = ['label', 'text'])
  return df

def preprocess(dataset):
  """
  Preprocesses the dataset

  arg: DataFrame dataset
  
  return: dataset
  """
  return dataset

def train_logistic_regression_model(df):
  """
  Trains Logistic Regression classifier

  arg: DataFrame dataset
  
  return: tuple of a trained,fitted model and a NLP vectorizer/transformer
  """
  text_train, text_test, label_train, label_test = train_test_split(df['text'], df['label_id'], test_size=0.15, random_state=10)

  tfidf = TfidfVectorizer(sublinear_tf=True, #scale the words frequency in logarithmic scale
                        min_df=5, #remove the words which has occurred in less than ‘min_df’ number of files
                        ngram_range=(1, 2), #dont know what role n-grams play in vectorisation
                        stop_words='english', #it removes stop words which are predefined in ‘english’.
                        lowercase=True #everything to lowercase
                        )
  features = tfidf.fit_transform(text_train).toarray()
  labels = label_train

  model = LogisticRegression(random_state=0, max_iter=400)
  model.fit(features, labels)

  features_test = tfidf.transform(text_test).toarray()
  label_pred = model.predict(features_test)

  return (model, tfidf)

def train_NB_classifier_model(df):
  """
  Trains Naive Bayes classifier

  arg: DataFrame dataset
  
  return: tuple of a trained,fitted model and a NLP vectorizer/transformer
  """
  text_train, text_test, label_train, label_test = train_test_split(df['text'], df['label_id'], test_size=0.15, random_state=10)

  tfidf = TfidfVectorizer(sublinear_tf=True, #scale the words frequency in logarithmic scale
                        min_df=5, #remove the words which has occurred in less than ‘min_df’ number of files
                        ngram_range=(1, 2), #dont know what role n-grams play in vectorisation
                        stop_words='english', #it removes stop words which are predefined in ‘english’.
                        lowercase=True #everything to lowercase
                        )
  features = tfidf.fit_transform(text_train).toarray()
  labels = label_train

  model = MultinomialNB()
  model.fit(features, labels)

  features_test = tfidf.transform(text_test).toarray()
  label_pred = model.predict(features_test)

  return (model, tfidf)

def classify_utterance_1(ut, dataset):
  """
  Classify the utterance as most frequent label in the dataset.

  arg:
    ut - the utterance
    dataset - DataFrame with dataset
    
  return: label of the utterance
  """
  classified_label = dataset['label'].mode()
  return classified_label


def classify_utterance_2(ut, dataset):
  """
  Classify the utterance according to the specific keywords it contains.

  arg:
    ut - the utterance
    dataset - DataFrame with dataset

  return: label of the utterance
  """
  if 'bye' in ut:
    classified_label = 'bye'
  elif 'no' in ut:
    classified_label = 'negate'
  elif 'start over' in ut:
    classified_label = 'restart'
  else:
    classified_label = 'null'
  return classified_label


def classify_utterance_3(ut, classifier, processor, labels):
  """
  Classify the utterance based on an ML model. for that purpose preprocess the utterance and feed it to model preditor.

  arg: 
    ut - utterance, 
    classifier - ML model, 
    processor - NLP vectorizer/transformer, 
    labels - label_id mapping

  return: label of the utterance
  """
  ready_ut = processor.transform([ut]).toarray()
  classified_label_id = classifier.predict(ready_ut)[0]
  classified_label = labels.loc[classified_label_id][0]
  return classified_label


def bot(model, utterance_processor, label_dict):
  """
  Runs a bot which (until 'bye123' is written)
  1.takes user input
  2.classifies the input text into an action label
  3.prints back out that label and text

  arg: 
    model - ML model for utterance classification
    utterance_processor - NLP vectorizer/processor
    label_dict - label_id decoder

  return: None
  """
  finished = False
  print(f'Hi! To exit enter "bye123"')
  while not finished:
    utterance = input('>').lower()
    if 'bye123' in utterance:
      finished = True
      print(f'Bye!')
      continue
    label = classify_utterance_3(utterance, model, utterance_processor, label_dict)
    print(f'utterance:{utterance}\nlabel:{label}')

def main():
  """
  Prepares the dataset, model and runs the bot
  """
  df = get_dataset('dialog_acts.dat')
  df = preprocess(df)
  df['label_id'] = df['label'].factorize()[0]
  label_dict = df[['label','label_id']].drop_duplicates().set_index('label_id')
  model, vectorizer = train_logistic_regression_model(df)
  #model, vectorizer = train_NB_classifier_model(df)
  bot(model, vectorizer, label_dict)

main()

