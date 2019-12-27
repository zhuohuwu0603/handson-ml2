#!/usr/bin/env python
# coding: utf-8

# # note
# * Didn't mean to generalize NN here. Just plow through this `400>25>10` setup to get the feeling of NN

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append('..')

from helper import nn
from helper import logistic_regression as lr
import numpy as np


# # prepare data

# In[2]:


X_raw, y_raw = nn.load_data('ex4data1.mat', transpose=False)
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)
X.shape


# ***

# In[3]:


y_raw


# In[4]:


y = nn.expand_y(y_raw)
y


# # load weight

# In[5]:


t1, t2 = nn.load_weight('ex4weights.mat')
t1.shape, t2.shape


# In[6]:


theta = nn.serialize(t1, t2)  # flatten params
theta.shape


# # feed forward
# > (400 + 1) -> (25 + 1) -> (10)
# 
# <img style="float: left;" src="../img/nn_model.png">

# In[7]:


_, _, _, _, h = nn.feed_forward(theta, X)
h # 5000*10


# # cost function
# <img style="float: left;" src="../img/nn_cost.png">

# think about this, now we have $y$ and $h_{\theta} \in R^{5000 \times 10}$  
# If you just ignore the m and k dimention, pairwisely this computation is trivial.  
# the eqation $= y*log(h_{\theta}) - (1-y)*log(1-h_{\theta})$  
# all you need to do after pairwise computation is sums this 2d array up and divided by m

# In[8]:


nn.cost(theta, X, y)


# # regularized cost function
# <img style="float: left;" src="../img/nn_regcost.png">

# the first column of t1 and t2 is intercept $\theta$, just forget them when you do regularization

# In[9]:


nn.regularized_cost(theta, X, y)


# In[ ]:




