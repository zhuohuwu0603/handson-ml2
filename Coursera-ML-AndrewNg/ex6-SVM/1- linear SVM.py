#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt


# In[2]:


mat = sio.loadmat('./data/ex6data1.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

data.head()


# # visualize data
# pay attention to the edge case at the left hand side

# In[3]:


fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['y'], cmap='Reds')
ax.set_title('Raw data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')


# # try $C=1$
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC

# In[4]:


svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
svc1.score(data[['X1', 'X2']], data['y'])


# In[5]:


data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])


# In[6]:


fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=1) Decision Confidence')


# # try $C=100$
# with large C, you try to overfit the data, so the left hand side edge case now is categorized right

# In[7]:


svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
svc100.score(data[['X1', 'X2']], data['y'])


# In[8]:


data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])


# In[9]:


fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM100 Confidence'], cmap='RdBu')
ax.set_title('SVM (C=100) Decision Confidence')


# In[10]:


data.head()


# In[ ]:




