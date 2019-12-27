#!/usr/bin/env python
# coding: utf-8

# In[11]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

from helper import nn


# In[12]:


X, y = nn.load_data('ex3data1.mat')

print(X.shape)
print(y.shape)


# ***

# In[13]:


pick_one = np.random.randint(0, 5000)
nn.plot_an_image(X[pick_one, :])
print('this should be {}'.format(y[pick_one]))


# In[14]:


nn.plot_100_image(X)


# In[ ]:




