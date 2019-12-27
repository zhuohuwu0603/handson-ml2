#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])


# In[10]:


df.head()


# In[11]:


df.info()


# ***
# # take a look at raw data

# In[12]:


sns.lmplot('population', 'profit', df, size=6, fit_reg=False)


# In[ ]:




