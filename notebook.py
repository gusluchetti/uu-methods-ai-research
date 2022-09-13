#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


with open('dialog_acts.dat', 'r') as f:
    data = f.readlines()
    data = list(map(lambda x: [x[0:x.index(" ")], x[x.index(" ")+1:-1]], data))


# In[3]:


df = pd.DataFrame(np.array(data), columns = ['label', 'text'])


# # Pre-Processing

# In[9]:


df['text'].describe()


# In[5]:


# if text contains "noise" or "tv_noise", remove it
# if that makes the column empty, remove the row
df_proc = df.copy()
print("before removing 'noise': ")
print(df_proc.describe())

for index, row in df_proc.iterrows():
    if "tv_noise" in row[1]:
        row[1] = row[1].replace('tv_noise', '')
    elif "noise" in row[1]:
        row[1] = row[1].replace('noise', '')
    
    row[1] = row[1].strip()
    
length = len(df_proc.index)
print(f'\nCurrent df length is {length}')

for i in range(length):
    text = df['text'][i]
    if not text:
        df_proc = df.drop([i])
        
print("\nafter: ")
print(df_proc.describe())


# In[10]:


df_proc['text'].describe()


# ## First Baseline System - Majority Label

# In[7]:


majority_label = df_proc['label'].mode()
print(majority_label)

