#!/usr/bin/env python
# coding: utf-8

# * [PEP 465 -- A dedicated infix operator for matrix multiplication](https://www.python.org/dev/peps/pep-0465/)

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')

from helper import linear_regression as lr  # my own module
from helper import general as general


# In[2]:


data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

data.head()


# # compute cost
# <img style="float: left;" src="../img/linear_cost.png">

# In[3]:


X = general.get_X(data)
print(X.shape, type(X))

y = general.get_y(data)
print(y.shape, type(y))


# In[4]:


theta = np.zeros(X.shape[1])


# In[5]:


lr.cost(theta, X, y)


# # batch gradient decent
# <img style="float: left;" src="../img/linear_gradient.png">

# In[6]:


epoch = 500
final_theta, cost_data = lr.batch_gradient_decent(theta, X, y, epoch)


# In[7]:


# compute final cost
lr.cost(final_theta, X, y)


# # visualize cost data

# In[8]:


ax = sns.tsplot(cost_data, time=np.arange(epoch+1))
ax.set_xlabel('epoch')
ax.set_ylabel('cost')


# In[9]:


b = final_theta[0] # intercept
m = final_theta[1] # slope

plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*m + b, label="Prediction")
plt.legend(loc=2)


# In[ ]:




