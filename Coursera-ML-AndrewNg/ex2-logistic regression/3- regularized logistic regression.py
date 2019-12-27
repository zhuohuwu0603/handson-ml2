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

import sys
sys.path.append('..')

# load my own module
from helper import logistic_regression as lr  
from helper import general as general

from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'accepted'])
df.head()


# # visualize data

# In[3]:


sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('test1', 'test2', hue='accepted', data=df, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 50}
          )

plt.title('Regularized Logistic Regression')


# # feature mapping
# 
# polynomial expansion
# 
# ```
# for i in 0..i
#   for p in 0..i:
#     output x^(i-p) * y^p
# ```
# <img style="float: left;" src="../img/mapped_feature.png">

# In[4]:


x1 = np.array(df.test1)
x2 = np.array(df.test2)


# In[5]:


data = lr.feature_mapping(x1, x2, power=6)
print(data.shape)
data.head()


# In[6]:


data.describe()


# # regularized cost
# <img style="float: left;" src="../img/reg_cost.png">

# In[7]:


theta = np.zeros(data.shape[1])
X = lr.feature_mapping(x1, x2, power=6, as_ndarray=True)
print(X.shape)

y = general.get_y(df)
print(y.shape)


# In[8]:


lr.regularized_cost(theta, X, y, l=1)


# this is the same as the not regularized cost because we init theta as zeros...

# # regularized gradient
# <img style="float: left;" src="../img/reg_gradient.png">

# In[9]:


lr.regularized_gradient(theta, X, y)


# # fit the parameters

# In[10]:


import scipy.optimize as opt


# In[11]:


print('init cost = {}'.format(lr.regularized_cost(theta, X, y)))

res = opt.minimize(fun=lr.regularized_cost, x0=theta, args=(X, y), method='Newton-CG', jac=lr.regularized_gradient)
res


# # predict

# In[12]:


final_theta = res.x
y_pred = lr.predict(X, final_theta)

print(classification_report(y, y_pred))

