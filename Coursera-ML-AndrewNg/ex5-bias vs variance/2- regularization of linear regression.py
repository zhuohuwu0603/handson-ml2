#!/usr/bin/env python
# coding: utf-8

# In[9]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('..')

from helper import linear_regression as lr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


X, y, Xval, yval, Xtest, ytest = lr.load_data()
# insert the intercept data of every X
X, Xval, Xtest = [np.insert(x.reshape(x.shape[0], 1), 0, np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


# # cost
# <img style="float: left;" src="../img/linear_cost.png">

# In[11]:


theta = np.ones(X.shape[1])
lr.cost(theta, X, y)


# # regularized cost
# <img style="float: left;" src="../img/linear_reg_cost.png">

# In[12]:


lr.regularized_cost(theta, X, y)


# # gradient
# <img style="float: left;" src="../img/linear_gradient.png">

# In[13]:


lr.gradient(theta, X, y)


# # regularized gradient
# <img style="float: left;" src="../img/linear_reg_gradient.png">

# In[14]:


lr.regularized_gradient(theta, X, y)


# # fit the data
# > regularization term $\lambda=0$

# In[15]:


theta = np.ones(X.shape[0])

final_theta = lr.linear_regression_np(X, y, l=0).get('x')


# In[16]:


b = final_theta[0] # intercept
m = final_theta[1] # slope

plt.scatter(X[:,1], y, label="Training data")
plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
plt.legend(loc=2)


# In[ ]:




