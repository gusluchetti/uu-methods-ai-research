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


# ## Models

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0, max_iter=400),
]


# In[ ]:


from sklearn.model_selection import cross_val_score

CV = 2
entries = []
cv_df = pd.DataFrame(index=range(CV * len(models)))

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy')
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[ ]:


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']

acc # seems like LogisticRegression is the best or we can tweek the hyperparameters more. 


# In[ ]:


features_test = tfidf.transform(text_test).toarray()


# In[ ]:


model = LogisticRegression(random_state=0, max_iter=400)
model.fit(features, labels)
label_pred = model.predict(features_test)


# In[ ]:





# ## Evaluations

# ### Baseline Systems

# In[ ]:





# ### Proper Models (Random Forest, Multinomial NB, Logistic Regression)

# In[17]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics

print(metrics.classification_report(label_test, label_pred, 
                                    target_names= df['label'].unique()))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt 

conf_mat = confusion_matrix(label_test, label_pred)
plt.figure(figsize = (20,5))
sns.heatmap(conf_mat, annot=True, cmap='Greens', fmt='d',
            xticklabels=label_dict.label.values, 
            yticklabels=label_dict.label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[ ]:


df.describe()


# In[ ]:


df.groupby('label').describe().sort_values(('text','count'))

