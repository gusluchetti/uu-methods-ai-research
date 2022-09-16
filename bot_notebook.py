#!/usr/bin/env python
# coding: utf-8

# # bot

# ## Your code should offer a prompt to enter a new utterance and classify this utterance, and repeat the prompt until the user exits.

# In[64]:


def classify_utterance_1(ut):
  classified_label = "inform"
  return classified_label

def classify_utterance_2(ut):
  if 'bye' in ut:
    classified_label = 'bye'
  elif 'no' in ut:
    classified_label = 'negate'
  elif 'start over' in ut:
    classified_label = 'restart'
  else:
    classified_label = 'null'

  return classified_label

def classify_utterance_3(ut):
  #classified_label = ml_lassifier_model.predict(ut)[0]
  classified_label_id = model.predict(tfidf.transform([ut]).toarray())[0]
  classified_label = label_dict.loc[classified_label_id][0]
  return classified_label

def bot():
  finished = False
  print(f'Hi! To exit enter "bye123"')
  while not finished:
    utterance = input('>').lower()
    label = classify_utterance_3(utterance)
    print(f'label:{label}\nutterance:{utterance}')  
    if 'bye123' in utterance:
      finished = True

