#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'notebook')

import sys
sys.path.append('..')

from helper import nn


# In[2]:


X, _ = nn.load_data('ex4data1.mat')


# In[3]:


nn.plot_100_image(X)


# In[ ]:




