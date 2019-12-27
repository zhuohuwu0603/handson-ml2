#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'notebook')

import sys
sys.path.append('..')

from helper import nn
from helper import logistic_regression as lr
import numpy as np


# # get data and weights

# In[2]:


X_raw, y_raw = nn.load_data('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
X.shape


# In[3]:


y = nn.expand_y(y_raw)
y.shape


# In[4]:


t1, t2 = nn.load_weight('ex4weights.mat')
t1.shape, t2.shape


# In[5]:


theta = nn.serialize(t1, t2)  # flatten params
theta.shape


# # sigmoid gradient

# In[6]:


nn.sigmoid_gradient(0)


# # theta gradient
# super hard to get this right... the dimension is so confusing

# In[7]:


d1, d2 = nn.deserialize(nn.gradient(theta, X, y))


# In[8]:


d1.shape, d2.shape


# # gradient checking
# <img style="float: left;" src="../img/gradient_checking.png">

# In[9]:


# nn.gradient_checking(theta, X, y, epsilon= 0.0001)


# # regularized gradient
# Use normal gradient + regularized term

# <img style="float: left;" src="../img/nn_reg_grad.png">

# In[10]:


# nn.gradient_checking(theta, X, y, epsilon=0.0001, regularized=True)


# # ready to train the model

# > remember to randomly initlized the parameters to break symmetry
# 
# take a look at the doc of this argument: `jac`
# 
# >jac : bool or callable, optional
# Jacobian (gradient) of objective function. Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg. **If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function.** If False, the gradient will be estimated numerically. jac can also be a callable returning the gradient of the objective. In this case, it must accept the same arguments as fun.
# 
# it means if your `backprop` function return `(cost, grad)`, you could set `jac=True`  
# 
# This is the implementation of http://nbviewer.jupyter.org/github/jdwittenauer/ipython-notebooks/blob/master/notebooks/ml/ML-Exercise4.ipynb
# 
# but I choose to seperate them

# In[11]:


res = nn.nn_training(X, y)
res


# # show accuracy

# In[12]:


_, y_answer = nn.load_data('ex4data1.mat')
y_answer[:20]


# In[13]:


final_theta = res.x


# In[16]:


nn.show_accuracy(final_theta, X, y_answer)


# # show hidden layer

# In[15]:


nn.plot_hidden_layer(final_theta)


# In[ ]:




