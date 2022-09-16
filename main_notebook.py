#!/usr/bin/env python
# coding: utf-8

# # WIP - Restaurant Recommendation Dialog System
# 
# ## Members:
# - Karpiński, R.R. (Rafał)
# - Pavan, L. (Lorenzo)
# - Rodrigues Luchetti, G.L. (Gustavo)
# - Teunissen, N.D. (Niels)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# In[2]:


def get_dataset(path):
    """
      args: dataset file path
      return: DataFrame of dataset
    """
    with open(path, 'r') as f:
        data = f.readlines()
        data = list(map(lambda x: x.rstrip("\n").split(" ", 1), data))
    df = pd.DataFrame(np.array(data), columns = ['label', 'text'])    
    return df


# In[3]:


def preprocess(dataset):
    """
    Preprocesses the dataset
    args: DataFrame dataset
    return: dataset
    """
    return dataset


# In[4]:


def label_factorize(df):
    """Does label factorization for a dataframe that has a column labeled 'label'"""
    # making label dict (turning labels into numbers)
    df['label_id'] = df['label'].factorize()[0]
    label_dict = df[['label','label_id']].drop_duplicates().set_index('label_id')
    
    return label_dict


# ## Building Baseline Systems

# In[5]:


def get_majority_class():
    majority_class = df['label'].mode().to_string(index = False)
    print(f"Majority class is '{majority_class}' ")
    
    return majority_class


# In[6]:


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


# In[7]:


def single_keyword_matching(text):
    """
    Rule-based prediction of a dialog act based on a phrase.
    args: utterance (any string)
    returns: Returns the predicted dialog act.
    """
    label = "inform"
    for key in keyword_dict:
        if (re.search(keyword_dict[key], text)): #if we find one of our keywords on any given string
            label = key
            return
    return label

def df_majority_class(dataframe):
    """
    Classifies dialog based on the majority class label.
    args: pandas DataFrame that contains a column named text with utterances.
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    predictions = []
    for i in range(0,len(dataframe)):
        predictions.append(majority_class)
    return predictions


# In[8]:


def single_majority_class(utterance):
    """
    Classifies dialog based on the majority class label.
    args: utterance (any string)
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    return majority_class      

def df_keyword_matching(dataframe):
    """
    Rule-based prediction of dialog acts based on a colletion of utterances.
    args: DataFrame that contains a column named text with utterances.
    returns: list of predictions about the label (dialog act) of the utterances.
    """
    predictions = []
    for i in range(0,len(dataframe)):
        text = df.loc[i, 'text']
        predictions.append(single_keyword_matching(text))
    return predictions


# ## Building Classifier Models

# In[9]:


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
        sublinear_tf=True, # scale the words frequency in logarithmic scale
        min_df=5, # remove the words which has occurred in less than ‘min_df’ number of files
        ngram_range=(1, 2), # don't know what role n-grams play in vectorisation
        stop_words='english', # it removes stop words which are predefined in ‘english’.
        lowercase=True # everything to lowercase
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    labels = label_train

    model = method
    model.fit(X_train_tfidf, y_train)

    X_test_tfidf = tfidf.transform(X_test).toarray()
    y_pred_test = model.predict(X_test_tfidf)

    return (model, tfidf)

# shorthands for model training
def train_logistic_regression_model(df):
    return train_model(LogisticRegression(random_state=0, max_iter=400), df)

def train_NB_classifier_model(df):
    return train_model(MultinomialNB(), df)


# In[11]:


#from sklearn.model_selection import cross_val_score

#CV = 5
#entries = []
#for model in models:
#    model_name = model.__class__.__name__
#    accuracies = cross_val_score(model, X_train_tfidf, y_train, scoring='accuracy')
#    for fold_idx, accuracy in enumerate(accuracies):
#        entries.append((model_name, fold_idx, accuracy))
#    
#cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[12]:


def main():
    """Prepares the dataset, model and runs the bot"""
    df = get_dataset('dialog_acts.dat')
    # df = preprocess(df)
    label_dict = label_factorize(df)
    majority_class = get_majority_class(df)
    
    model, vectorizer = train_logistic_regression_model(df)
    # model, vectorizer = train_NB_classifier_model(df)
    bot(model, vectorizer, label_dict)


# In[13]:


# main()


# ---

# ---

# ---

# ## Evaluations

# In[14]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# ### Baseline Systems

# In[15]:


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


# In[16]:


predictions = df_keyword_matching(df)
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


# In[ ]:


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


# ### Proper Models (Random Forest, Multinomial NB, Logistic Regression)

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

