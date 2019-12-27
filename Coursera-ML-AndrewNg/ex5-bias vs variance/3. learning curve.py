#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('..')

from helper import linear_regression as lr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


X, y, Xval, yval, Xtest, ytest = lr.load_data()
# insert the intercept data of every X
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


# In[3]:


training_cost, cv_cost = [], []


# 1. use the subset of training set to fit the model
# 2. no regularization when you compute the training cost and CV cost
# 3. remember to use the same subset of training set to compute training cost

# In[4]:


m = X.shape[0]
for i in range(1, m+1):
#     print('i={}'.format(i))
    res = lr.linear_regression_np(X[:i, :], y[:i], l=0)
    
    tc = lr.regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = lr.regularized_cost(res.x, Xval, yval, l=0)
#     print('tc={}, cv={}'.format(tc, cv))
    
    training_cost.append(tc)
    cv_cost.append(cv)


# In[5]:


plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend(loc=1)


# the model can't even fit traning set well, **under fitting**

# In[ ]:




