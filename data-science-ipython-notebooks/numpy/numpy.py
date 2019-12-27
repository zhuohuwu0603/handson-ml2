#!/usr/bin/env python
# coding: utf-8

# # NumPy
# 
# Credits: Forked from [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
# 
# * NumPy Arrays, dtype, and shape
# * Common Array Operations
# * Reshape and Update In-Place
# * Combine Arrays
# * Create Sample Data

# In[1]:


import numpy as np


# ## NumPy Arrays, dtypes, and shapes

# In[2]:


a = np.array([1, 2, 3])
print(a)
print(a.shape)
print(a.dtype)


# In[3]:


b = np.array([[0, 2, 4], [1, 3, 5]])
print(b)
print(b.shape)
print(b.dtype)


# In[4]:


np.zeros(5)


# In[5]:


np.ones(shape=(3, 4), dtype=np.int32)


# ## Common Array Operations

# In[6]:


c = b * 0.5
print(c)
print(c.shape)
print(c.dtype)


# In[7]:


d = a + c
print(d)


# In[8]:


d[0]


# In[9]:


d[0, 0]


# In[10]:


d[:, 0]


# In[11]:


d.sum()


# In[12]:


d.mean()


# In[13]:


d.sum(axis=0)


# In[14]:


d.mean(axis=1)


# ## Reshape and Update In-Place

# In[15]:


e = np.arange(12)
print(e)


# In[16]:


# f is a view of contents of e
f = e.reshape(3, 4)
print(f)


# In[17]:


# Set values of e from index 5 onwards to 0
e[5:] = 0
print(e)


# In[18]:


# f is also updated
f


# In[19]:


# OWNDATA shows f does not own its data
f.flags


# ## Combine Arrays

# In[20]:


a


# In[21]:


b


# In[22]:


d


# In[23]:


np.concatenate([a, a, a])


# In[24]:


# Use broadcasting when needed to do this automatically
np.vstack([a, b, d])


# In[25]:


# In machine learning, useful to enrich or 
# add new/concatenate features with hstack
np.hstack([b, d])


# ## Create Sample Data

# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pylab as plt
import seaborn

seaborn.set()


# In[27]:


# Create evenly spaced numbers over the specified interval
x = np.linspace(0, 2, 10)
plt.plot(x, 'o-');
plt.show()


# In[28]:


# Create sample data, add some noise
x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)

plt.scatter(x, y)
plt.show()

