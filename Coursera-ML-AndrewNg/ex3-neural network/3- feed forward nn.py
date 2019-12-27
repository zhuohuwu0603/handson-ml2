#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np

import sys
sys.path.append('..')

from helper import nn
from helper import logistic_regression as lr

from sklearn.metrics import classification_report


# # model
# <img style="float: left;" src="../img/nn_model.png">

# # load weights and data

# In[2]:


theta1, theta2 = nn.load_weight('ex3weights.mat')

theta1.shape, theta2.shape


# >The original data is 90 degree off. So in data loading function, I use transpose to fix it.  
# However, the transposed data is not compatible to given parameters because those parameters are trained by original data. So for the sake of applying given parameters, I need to use original data

# In[3]:


X, y = nn.load_data('ex3data1.mat',transpose=False)

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

X.shape, y.shape


# # feed forward prediction

# In[4]:


a1 = X


# In[5]:


z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
z2.shape


# In[6]:


z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)


# In[7]:


a2 = lr.sigmoid(z2)
a2.shape


# In[8]:


z3 = a2 @ theta2.T
z3.shape


# In[9]:


a3 = lr.sigmoid(z3)
a3


# In[10]:


y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention
y_pred.shape


# # accuracy
# so... accuracy on training data is not predicting the real world performance you know  
# All we can say is NN is very powerful model. Overfitting is easy here.

# In[11]:


print(classification_report(y_pred, y_pred))


# In[ ]:




