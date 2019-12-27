#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import scipy.io as sio


# > I think the hard part is how to vecotrize emails.  
# Using this preprocessed data set is cheating XD

# In[2]:


mat_tr = sio.loadmat('data/spamTrain.mat')
mat_tr.keys()


# > be careful with the column vector : `(4000, 1)` is not the same as `(4000, )`

# In[3]:


X, y = mat_tr.get('X'), mat_tr.get('y').ravel()
X.shape, y.shape


# In[4]:


mat_test = sio.loadmat('data/spamTest.mat')
mat_test.keys()


# In[5]:


test_X, test_y = mat_test.get('Xtest'), mat_test.get('ytest').ravel()
test_X.shape, test_y.shape


# # fit SVM model

# In[6]:


svc = svm.SVC()


# In[7]:


svc.fit(X, y)


# In[8]:


pred = svc.predict(test_X)
print(metrics.classification_report(test_y, pred))


# # what about linear logistic regresion?

# In[9]:


logit = LogisticRegression()
logit.fit(X, y)


# In[10]:


pred = logit.predict(test_X)
print(metrics.classification_report(test_y, pred))


# .......... then what for.... SVM
