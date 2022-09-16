#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0, max_iter=400),
]


# In[9]:


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


from sklearn.metrics import confusion_matrix
from sklearn import metrics

print(metrics.classification_report(label_test, label_pred, 
                                    target_names= df['label'].unique()))


# ## Plots

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt 

conf_mat = confusion_matrix(label_test, label_pred)
plt.figure(figsize = (20,5))
sns.heatmap(conf_mat, annot=True, cmap='Greens', fmt='d',
            xticklabels=label_dict.label.values, 
            yticklabels=label_dict.label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[11]:


df.describe()


# In[ ]:


df.groupby('label').describe().sort_values(('text','count'))

