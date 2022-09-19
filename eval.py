#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import cross_val_score

CV = 5
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train_tfidf, y_train, scoring='accuracy')
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# ## Evaluations

# In[1]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# ### Baseline Systems

# In[3]:


def plot_confusion_matrix(labels,predictions):
    """Plots the confusion matrix
    Arguments:
    labels: array-like of shape (n_samples,)
    predictions: array-like of shape (n_samples,)
    Returns
    -------
    plot
        plots the confusion matrix
    """
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams['font.size'] = 8
    
    ConfusionMatrixDisplay.from_predictions(labels,predictions)
    plt.show()


# In[4]:


# predictions = df_keyword_matching(df)
def baselineAccuracy(predictions, df):
    """Calculates the accuracy
        Arguments:
        predictions: list
        df: a pandas dataframe that contains a column named text with utterances.
        Returns
        -------
    Returns:
        Returns the accuracy
    """
    count = 0
    for i in range(0,len(predictions)):
        
        if(predictions[i].lower() == df.loc[i,'label'].lower()):
            count += 1
    return "Accuracy: "+str(round(count / len(predictions)*100,1))+"%"


# In[5]:


baselineAccuracy(predictions, df)


# In[ ]:


def metrics_overview(labels, predictions):
       """Prints metrics
       Arguments:
       labels: array-like of shape (n_samples,)
       predictions: array-like of shape (n_samples,)
       
       Prints different metrics related to the confusion matrix.
       """
       edges_confusion_matrix = confusion_matrix(labels,predictions)

       FP = edges_confusion_matrix.sum(axis=0) - np.diag(edges_confusion_matrix)  
       
       FN = edges_confusion_matrix.sum(axis=1) - np.diag(edges_confusion_matrix)
       
       TP = np.diag(edges_confusion_matrix)
       
       TN = edges_confusion_matrix.sum() - (FP + FN + TP)
       
       
       # Sensitivity, hit rate, recall, or true positive rate
       TPR = TP/(TP+FN)
       print('TPR',TPR)
       print('Average TPR',np.average(TPR))
       print('_______________________________')
       # Specificity or true negative rate
       TNR = TN/(TN+FP)
       print('TNR',TNR)
       print('Average TNR',np.average(TNR))
       print('_______________________________')

       # Precision or positive predictive value
       PPV = TP/(TP+FP)
       print('PPV',PPV)
       print('Average PPV',np.average(PPV))
       print('_______________________________')

       # Negative predictive value
       NPV = TN/(TN+FN)
       print('NPV',NPV)
       print('Average NPV',np.average(NPV))
       print('_______________________________')

       # Fall out or false positive rate
       FPR = FP/(FP+TN)
       print('FPR',FPR)
       print('Average FPR',np.average(FPR))
       print('_______________________________')

       # False negative rate
       FNR = FN/(TP+FN)
       print('FNR',FNR)
       print('Average FNR',np.average(FNR))
       print('_______________________________')

       # False discovery rate
       FDR = FP/(TP+FP)
       print('FDR',FDR)
       print('Average FDR',np.average(FDR))
       print('_______________________________')

       # Overall accuracy
       ACC = (TP+TN)/(TP+FP+FN+TN)
       print('ACC',ACC)
       print('Average ACC',np.average(ACC))
       print('_______________________________')

       F1 = 2*((PPV*TPR)/(PPV+TPR))
       F1 = F1[~np.isnan(F1)]
       print('F1',F1)
       print('Average F1',np.average(F1))
       print('_______________________________')
       print((FP+FN)/(TP+FP+FN+TN))
       
metrics_overview(df['label'],predictions)


# In[ ]:


### Proper Models (Random Forest, Multinomial NB, Logistic Regression)


# In[ ]:


mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, 
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']

acc


# In[ ]:


print(classification_report(y_test, y_pred_test, target_names= df['label'].unique()))


# In[ ]:


import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize = (20,5))
plt.ylabel('Actual')
plt.xlabel('Predicted')

sns.heatmap(conf_mat, annot=True, cmap='Greens', fmt='d',
            xticklabels=label_dict.label.values, 
            yticklabels=label_dict.label.values)


# In[ ]:


conf_mat = confusion_matrix(df['label'], predictions)
plt.figure(figsize = (20,5))
plt.ylabel('Actual')
plt.xlabel('Predicted')

sns.heatmap(conf_mat, annot=True, cmap='Greens', fmt='d',
            xticklabels=label_dict.label.values, 
            yticklabels=label_dict.label.values)


# In[ ]:


df.groupby('label').describe()

