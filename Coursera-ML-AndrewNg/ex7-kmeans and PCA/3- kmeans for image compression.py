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

import sys
sys.path.append('..')

from helper import kmeans as km


# # load image as `ndarray`
# http://scikit-image.org/

# In[2]:


from skimage import io

# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('data/bird_small.png') / 255.
io.imshow(pic)


# In[3]:


pic.shape


# In[12]:


# serialize data
data = pic.reshape(128*128, 3)


# # let's do k-mean

# * my version will take more than 10 mins... ok. I know why I shouldn't implement my own ML library.
# 
# * In the future I will only implement ML algorithm for the sake of learning it XD

# In[5]:


# C, centroids, cost = km.k_means(pd.DataFrame(data), 16, epoch = 10, n_init=3)


# # sklearn KMeans

# In[6]:


from sklearn.cluster import KMeans

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)


# In[7]:


model.fit(data)


# In[8]:


centroids = model.cluster_centers_
print(centroids.shape)

C = model.predict(data)
print(C.shape)


# In[9]:


centroids[C].shape


# In[10]:


compressed_pic = centroids[C].reshape((128,128,3))


# In[11]:


fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)

