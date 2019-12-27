#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import scipy.io as sio

import sys
sys.path.append('..')

from helper import kmeans as km


# In[2]:


mat = sio.loadmat('./data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
print(data2.head())

sns.set(context="notebook", style="white")
sns.lmplot('X1', 'X2', data=data2, fit_reg=False)


# # 0. random init
# for initial centroids

# In[3]:


km.random_init(data2, 3)


# # 1. cluster assignment
# http://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point

# ### find closest cluster experiment

# In[4]:


init_centroids = km.random_init(data2, 3)
init_centroids


# In[5]:


x = np.array([1, 1])


# In[6]:


fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x=init_centroids[:, 0], y=init_centroids[:, 1])

for i, node in enumerate(init_centroids):
    ax.annotate('{}: ({},{})'.format(i, node[0], node[1]), node)
    
ax.scatter(x[0], x[1], marker='x', s=200)


# In[7]:


km._find_your_cluster(x, init_centroids)


# ### 1 epoch cluster assigning

# In[8]:


C = km.assign_cluster(data2, init_centroids)
data_with_c = km.combine_data_C(data2, C)
data_with_c.head()


# See the first round clustering result

# In[9]:


sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)


# # 2. calculate new centroid

# In[10]:


km.new_centroids(data2, C)


# # putting all together, take1
# this is just 1 shot `k-means`, if the random init pick the bad starting centroids, the final clustering may be very sub-optimal

# In[11]:


final_C, final_centroid, _= km._k_means_iter(data2, 3)
data_with_c = km.combine_data_C(data2, final_C)


# In[12]:


sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)


# # calculate the cost

# In[13]:


km.cost(data2, final_centroid, final_C)


# # k-mean with multiple tries of randome init, pick the best one with least cost

# In[14]:


best_C, best_centroids, least_cost = km.k_means(data2, 3)


# In[15]:


least_cost


# In[16]:


data_with_c = km.combine_data_C(data2, best_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)


# # try sklearn kmeans

# In[17]:


from sklearn.cluster import KMeans


# In[18]:


sk_kmeans = KMeans(n_clusters=3)


# In[19]:


sk_kmeans.fit(data2)


# In[20]:


sk_C = sk_kmeans.predict(data2)


# In[21]:


data_with_c = km.combine_data_C(data2, sk_C)
sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)


# In[ ]:




