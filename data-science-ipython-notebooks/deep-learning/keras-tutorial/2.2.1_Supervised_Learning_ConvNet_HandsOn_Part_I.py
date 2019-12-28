#!/usr/bin/env python
# coding: utf-8

# Credits: Forked from [deep-learning-keras-tensorflow](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio

# # ConvNet HandsOn with Keras

# ## Problem Definition
# 
# *Recognize handwritten digits*

# ## Data
# 
# The MNIST database ([link](http://yann.lecun.com/exdb/mnist)) has a database of handwritten digits. 
# 
# The training set has $60,000$ samples. 
# The test set has $10,000$ samples.
# 
# The digits are size-normalized and centered in a fixed-size image. 
# 
# The data page has description on how the data was collected. It also has reports the benchmark of various algorithms on the test dataset. 

# ### Load the data
# 
# The data is available in the repo's `data` folder. Let's load that using the `keras` library. 
# 
# For now, let's load the data and see how it looks.

# In[1]:


import numpy as np
import keras
from keras.datasets import mnist


# In[2]:


# get_ipython().system('mkdir -p $HOME/.keras/datasets/euroscipy_2016_dl-keras/data/')


# In[4]:


# Set the full path to mnist.pkl.gz
path_to_dataset = "euroscipy_2016_dl-keras/data/mnist.pkl.gz"


# In[5]:


# Load the datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data(path_to_dataset)


# # Basic data analysis on the dataset

# In[6]:


# What is the type of X_train?


# In[8]:


# What is the type of y_train?


# In[9]:


# Find number of observations in training data


# In[10]:


# Find number of observations in test data


# In[23]:


# Display first 2 records of X_train


# In[24]:


# Display the first 10 records of y_train


# In[26]:


# Find the number of observations for each digit in the y_train dataset 


# In[27]:


# Find the number of observations for each digit in the y_test dataset 


# In[5]:


# What is the dimension of X_train?. What does that mean?


# ### Display Images
# 
# Let's now display some of the images and see how they look
# 
# We will be using `matplotlib` library for displaying the image

# In[11]:


from matplotlib import pyplot
import matplotlib as mpl
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


# Displaying the first training data


# In[4]:


fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
imgplot = ax.imshow(X_train[1], cmap=mpl.cm.Greys)
imgplot.set_interpolation('nearest')
ax.xaxis.set_ticks_position('top')
ax.yaxis.set_ticks_position('left')
pyplot.show()


# In[ ]:


# Let's now display the 11th record


# In[52]:




