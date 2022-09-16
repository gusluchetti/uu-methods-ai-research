#!/usr/bin/env python
# coding: utf-8

# In[270]:


with open('dialog_acts.dat', 'r') as f:
    data = f.readlines()
    data = list(map(lambda x: [x[0:x.index(" ")], x[x.index(" ")+1:-1]], data))
df = pd.DataFrame(np.array(data), columns = ['label', 'text'])
df = df[df.label != 'null']
df = df.reset_index()


# In[288]:





# In[289]:





# In[290]:





# In[291]:





# In[292]:





# In[293]:





# In[294]:




