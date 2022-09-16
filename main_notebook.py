#!/usr/bin/env python
# coding: utf-8

# # Text Classification Assignment

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

from re import search


# ## Reading 'dialog_acts.dat' Dataset

# In[2]:


with open('dialog_acts.dat', 'r') as f:
    data = f.readlines()
    data = list(map(lambda x: x.rstrip("\n").split(" ", 1), data))
    
df = pd.DataFrame(np.array(data), columns = ['label', 'text'])


# In[3]:


df.head(3)


# ## Pre-Processing
# 
# Looking for null values, irrelevant or noisy text (literally, removing 'tv_noise' and 'noise') and repeated values. Formatting labels into numbers.

# In[18]:


df['label_id'] = df['label'].factorize()[0]
label_dict = df[['label','label_id']].drop_duplicates().set_index('label_id')

label_dict


# In[8]:


from sklearn.model_selection import train_test_split
# X - independent features (excluding target variable).
# y - dependent variables (target we're looking to predict).

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_id'], test_size=0.15, random_state=10
)


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, # scale the words frequency in logarithmic scale
                        min_df=5, # remove the words which has occurred in less than ‘min_df’ number of files
                        ngram_range=(1, 2), # don't know what role n-grams play in vectorisation
                        stop_words='english', # it removes stop words which are predefined in ‘english’.
                        lowercase=True # everything to lowercase
                        )

features = tfidf.fit_transform(text_train).toarray()
targets = label_train

print(features)
print(targets)


# ## Evaluations

# ### Baseline Systems

# In[ ]:





# ### Proper Models (Random Forest, Multinomial NB, Logistic Regression)

# In[17]:




