import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ttest_rel

data = pd.read_excel(r"C:\Users\niels\Downloads\cleanedMAIR.xlsx")
data.head()
data['average_humanness'] = ((data.l1 + data.l2 + data.l3 + data.l4 + data.l5 + data.l6) / 6)
data['average_satisfaction'] = ((data.cs1 + data.cs2 + data.cs3) / 3)
data['Experiment_type'] = np.where(data['experiment_group'] == 'A', 'Delay', 'No Delay')

print(data.head())

average_humanness_nodelay = data['average_humanness'].where(data['Experiment_type'] == 'No Delay').dropna().mean()
average_humanness_delay = data['average_humanness'].where(data['Experiment_type'] == 'Delay').dropna().mean()

average_satisfaction_nodelay = data['average_satisfaction'].where(data['Experiment_type'] == 'No Delay').dropna().mean()
average_satisfaction_delay = data['average_satisfaction'].where(data['Experiment_type'] == 'Delay').dropna().mean()


barplot=plt.bar(x=data.Experiment_type.unique(), height=[average_humanness_nodelay,average_humanness_delay])
plt.bar_label(barplot,labels=[average_humanness_nodelay,average_humanness_delay],label_type='edge')
plt.title('Average humanness (0-9 semantic differential scale) in the different conditions')
plt.show()


# In[87]:


barplot=plt.bar(x=data.Experiment_type.unique(), height=[average_satisfaction_nodelay,average_satisfaction_delay])
plt.bar_label(barplot,labels=[average_satisfaction_nodelay,average_satisfaction_delay],label_type='edge')
plt.title('Average satisfaction (1-7 Likert scale) in the different conditions')
plt.show()


# In[88]:


x, y = data['average_humanness'], data['average_satisfaction']
plt.scatter(x,y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),color='red')
corr, _ = pearsonr(x, y)
print('Pearsons correlation: %.3f' % corr)
plt.xlabel('Average humanness')
plt.ylabel('Average satisfaction')
plt.title('Average humanness and satisfaction')
plt.show()


# In[89]:


average_humanness_nodelay2 = data['average_humanness'].where(data['Experiment_type'] == 'No Delay').dropna()
average_humanness_delay2 = data['average_humanness'].where(data['Experiment_type'] == 'Delay').dropna()

average_satisfaction_nodelay2 = data['average_satisfaction'].where(data['Experiment_type'] == 'No Delay').dropna()
average_satisfaction_delay2 = data['average_satisfaction'].where(data['Experiment_type'] == 'Delay').dropna()



x, y = average_humanness_nodelay2, average_satisfaction_nodelay2
plt.scatter(x,y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),color='red')
corr, _ = pearsonr(x, y)
print('Pearsons correlation: %.3f' % corr)
plt.xlabel('Average humanness')
plt.ylabel('Average satisfaction')
plt.title('Average humanness and satisfaction (No delay)')
plt.show()



# In[90]:


x, y = average_humanness_delay2, average_satisfaction_delay2
plt.scatter(x,y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),color='red')
corr, _ = pearsonr(x, y)
print('Pearsons correlation: %.3f' % corr)
plt.xlabel('Average humanness')
plt.ylabel('Average satisfaction')
plt.title('Average humanness and satisfaction (Delay)')
plt.show()


# In[95]:


ttest_humanness= ttest_rel(average_humanness_delay2,average_humanness_nodelay2)
ttest_satisfaction = ttest_rel(average_satisfaction_delay2,average_satisfaction_nodelay2)


# In[96]:


print(ttest_humanness)
print(ttest_satisfaction)


# In[ ]:




