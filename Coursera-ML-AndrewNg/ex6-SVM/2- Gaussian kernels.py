#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn import svm

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio


# In[2]:


# kernek function
def gaussian_kernel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))


# In[3]:


x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

gaussian_kernel(x1, x2, sigma)


# # load data

# In[4]:


mat = sio.loadmat('./data/ex6data2.mat')
print(mat.keys())
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

data.head()


# In[5]:


data.shape


# # visualize data

# In[6]:


sns.set(context="notebook", style="white", palette=sns.diverging_palette(240, 10, n=2))
sns.lmplot('X1', 'X2', hue='y', data=data, 
           size=5, 
           fit_reg=False, 
           scatter_kws={"s": 10}
          )


# # try built-in Gaussian Kernel of sklearn

# In[7]:


svc = svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)
svc


# In[8]:


svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])


# the `predict_proba` will give you ndarray (data size, class)  
# so if you just want to plot the decision contour of this binary example, choose one class and plot it 

# In[9]:


predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]


# In[10]:


fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')


# In[ ]:




