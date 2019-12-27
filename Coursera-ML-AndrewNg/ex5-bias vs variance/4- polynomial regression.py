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


# load in raw data, not intercept term
X, y, Xval, yval, Xtest, ytest = lr.load_data()


# # create polynomial features

# In[3]:


lr.poly_features(X, power=3)


# # prepare polynomial regression data
# 1. expand feature with power = 8, or any power you want to use 
# 2. Apply **normalization** to combat $x^n$ situation
# 3. don't forget intercept term

# In[4]:


X_poly, Xval_poly, Xtest_poly= lr.prepare_poly_data(X, Xval, Xtest, power=8)
X_poly[:3, :]


# # plot learning curve
# > again, first we don't apply regularization $\lambda=0$

# In[5]:


lr.plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)


# as you can see the training cost is too low to be true. This is **over fitting**

# # try $\lambda=1$

# In[6]:


lr.plot_learning_curve(X_poly, y, Xval_poly, yval, l=1)


# training cost increat a little bit, not 0 anymore.  
# Say we alleviate **over fitting** a little bit

# # try $\lambda=100$

# In[7]:


lr.plot_learning_curve(X_poly, y, Xval_poly, yval, l=100)


# too much regularization.  
# back to **underfit** situation

# # find the best $\lambda$

# In[8]:


l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []


# In[9]:


for l in l_candidate:
    res = lr.linear_regression_np(X_poly, y, l)
    
    tc = lr.cost(res.x, X_poly, y)
    cv = lr.cost(res.x, Xval_poly, yval)
    
    training_cost.append(tc)
    cv_cost.append(cv)


# In[10]:


plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)

plt.xlabel('lambda')

plt.ylabel('cost')


# In[11]:


# best cv I got from all those candidates
l_candidate[np.argmin(cv_cost)]


# In[12]:


# use test data to compute the cost
for l in l_candidate:
    theta = lr.linear_regression_np(X_poly, y, l).x
    print('test cost(l={}) = {}'.format(l, lr.cost(theta, Xtest_poly, ytest)))


# turns out $\lambda = 0.3$ is even better choice XD

# In[ ]:




