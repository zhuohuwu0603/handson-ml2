#!/usr/bin/env python
# coding: utf-8

# # notes
# * from now on I will focus on using `seaborn`. should be more than enoough for simple EDA purpose

# In[1]:


import pandas as pd
import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')


# # prepare data

# In[2]:


df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print(df.shape)
df.head()


# In[3]:


df.describe()


# # seaborn

# In[7]:


sns.set(context="notebook", style="darkgrid", palette=sns.color_palette("RdBu", 2))

sns.lmplot('exam1', 'exam2', hue='admitted', data=df, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 50}
          )


# In[ ]:




