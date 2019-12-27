#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white")

import numpy as np
import pandas as pd
import scipy.io as sio

import sys
sys.path.append('..')

from helper import general
from helper import pca


# # load data

# In[6]:


mat = sio.loadmat('./data/ex7data1.mat')
X = mat.get('X')

# visualize raw data
print(X.shape)

sns.lmplot('X1', 'X2', 
           data=pd.DataFrame(X, columns=['X1', 'X2']),
           fit_reg=False)


# # normalize data

# In[3]:


X_norm = pca.normalize(X)

sns.lmplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False)


# # covariance matrix $\Sigma$
# <img style="float: left;" src="../img/cov_mat.png">

# this is biased sample covariance matrix, for unbiased version, you need to divide it by $m-1$

# In[4]:


Sigma = pca.covariance_matrix(X_norm)  # capital greek Sigma
Sigma  # (n, n)


# # PCA
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html

# In[12]:


U, S, V = pca.pca(X_norm)


# In[13]:


U


# In[7]:


u1 = U[0]
u1


# # project data to lower dimension

# In[8]:


# show top 10 projected data
Z = pca.project_data(X_norm, U, 1)
Z[:10]


# http://stackoverflow.com/a/23973562/3943702

# In[16]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax1)
ax1.set_title('Original dimension')

sns.rugplot(Z, ax=ax2)
ax2.set_xlabel('Z')
ax2.set_title('Z dimension')


# # recover data to original dimension
# Of course, there would be inevitable information loss if you boost data from lower to higher dimension

# In[17]:


X_recover = pca.recover_data(Z, U)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))

sns.rugplot(Z, ax=ax1)
ax1.set_title('Z dimension')
ax1.set_xlabel('Z')

sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_recover, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax2)
ax2.set_title("2D projection from Z")

sns.regplot('X1', 'X2', 
           data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
           fit_reg=False,
           ax=ax3)
ax3.set_title('Original dimension')


# ### the projection from `(X1, X2)` to `Z` could be visualized like this

# <img style="float: central;" src="../img/pca_projection.png">

# In[ ]:




