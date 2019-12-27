#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import pandas as pd
import scipy.io as sio


# # data1

# In[2]:


mat = sio.loadmat('./data/ex7data1.mat')
mat.keys()


# In[3]:


data1 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data1.head()


# ***

# In[4]:


sns.set(context="notebook", style="white")


# In[5]:


sns.lmplot('X1', 'X2', data=data1, fit_reg=False)


# # data2

# In[6]:


mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data2.head()


# In[7]:


sns.lmplot('X1', 'X2', data=data2, fit_reg=False)


# In[ ]:




