#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="darkgrid", palette="dark")

import sys
sys.path.append('..')

from helper import linear_regression as lr  # my own module
from helper import general as general


# In[2]:


raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
raw_data.head()


# # 1. normalize data

# In[3]:


data = general.normalize_feature(raw_data)
data.head()


# # 2. multi-var batch gradient decent

# In[4]:


X = general.get_X(data)
print(X.shape, type(X))

y = general.get_y(data)
print(y.shape, type(y))


# In[5]:


alpha = 0.01
theta = np.zeros(X.shape[1])
epoch = 500


# In[6]:


final_theta, cost_data = lr.batch_gradient_decent(theta, X, y, epoch, alpha=alpha)


# In[11]:


sns.tsplot(time=np.arange(len(cost_data)), data = cost_data)


# In[8]:


final_theta


# # 3. learning rate

# In[9]:


base = np.logspace(-1, -5, num=4)
candidate = np.sort(np.concatenate((base, base*3)))
print(candidate)


# In[10]:


epoch=50

fig, ax = plt.subplots(figsize=(16, 9))

for alpha in candidate:
    _, cost_data = lr.batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
    ax.plot(np.arange(epoch+1), cost_data, label=alpha)

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)


# # 4. normal equation
# > this is fancy, but I don't see how this could help for non-convex optimization... so skip this one

# In[ ]:




