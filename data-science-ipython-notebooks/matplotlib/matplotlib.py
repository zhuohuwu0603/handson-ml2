#!/usr/bin/env python
# coding: utf-8

# # matplotlib

# Credits: Content forked from [Parallel Machine Learning with scikit-learn and IPython](https://github.com/ogrisel/parallel_ml_tutorial) by Olivier Grisel
# 
# * Setting Global Parameters
# * Basic Plots
# * Histograms
# * Two Histograms on the Same Plot
# * Scatter Plots

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import pylab as plt
import seaborn


# ## Setting Global Parameters

# In[2]:


# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Set seaborn aesthetic parameters to defaults
seaborn.set()


# ## Basic Plots

# In[3]:


x = np.linspace(0, 2, 10)

plt.plot(x, x, 'o-', label='linear')
plt.plot(x, x ** 2, 'x-', label='quadratic')

plt.legend(loc='best')
plt.title('Linear vs Quadratic progression')
plt.xlabel('Input')
plt.ylabel('Output');
plt.show()


# ## Histograms

# In[4]:


# Gaussian, mean 1, stddev .5, 1000 elements
samples = np.random.normal(loc=1.0, scale=0.5, size=1000)
print(samples.shape)
print(samples.dtype)
print(samples[:30])
plt.hist(samples, bins=50);
plt.show()


# ## Two Histograms on the Same Plot

# In[5]:


samples_1 = np.random.normal(loc=1, scale=.5, size=10000)
samples_2 = np.random.standard_t(df=10, size=10000)
bins = np.linspace(-3, 3, 50)

# Set an alpha and use the same bins since we are plotting two hists
plt.hist(samples_1, bins=bins, alpha=0.5, label='samples 1')
plt.hist(samples_2, bins=bins, alpha=0.5, label='samples 2')
plt.legend(loc='upper left');
plt.show()


# ## Scatter Plots

# In[6]:


plt.scatter(samples_1, samples_2, alpha=0.1);
plt.show()

