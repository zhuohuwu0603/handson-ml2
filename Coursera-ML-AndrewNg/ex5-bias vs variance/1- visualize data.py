#!/usr/bin/env python
# coding: utf-8

# In[4]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'notebook')

import sys
sys.path.append('..')

from helper import linear_regression as lr

import pandas as pd
import seaborn as sns


# In[5]:


X, y, Xval, yval, Xtest, ytest = lr.load_data()


# In[6]:


df = pd.DataFrame({'water_level':X, 'flow':y})

sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)


# In[ ]:




