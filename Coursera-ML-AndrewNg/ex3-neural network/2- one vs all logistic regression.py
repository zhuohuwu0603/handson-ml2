#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np

import sys
sys.path.append('..')

from helper import nn
from helper import logistic_regression as lr

from sklearn.metrics import classification_report


# In[2]:


raw_X, raw_y = nn.load_data('ex3data1.mat')
print(raw_X.shape)
print(raw_y.shape)


# # prepare data

# In[3]:


# add intercept=1 for x0
X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)
X.shape


# In[4]:


# y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
# I'll ditit 0, index 0 again
y_matrix = []

for k in range(1, 11):
    y_matrix.append((raw_y == k).astype(int))

# last one is k==10, it's digit 0, bring it to the first position
y_matrix = [y_matrix[-1]] + y_matrix[:-1]
y = np.array(y_matrix)

y.shape


# # train 1 model

# In[5]:


t0 = lr.logistic_regression(X, y[0])


# In[6]:


print(t0.shape)
y_pred = lr.predict(X, t0)
print('Accuracy={}'.format(np.mean(y[0] == y_pred)))


# Is this real......

# # train k model

# In[7]:


k_theta = np.array([lr.logistic_regression(X, y[k]) for k in range(10)])
print(k_theta.shape)


# # making prediction
# * think about the shape of k_theta, now you are making $X\times\theta^T$
# > $(5000, 401) \times (10, 401).T = (5000, 10)$
# * after that, you run sigmoid to get probabilities and for each row, you find the highest prob as the answer

# In[8]:


prob_matrix = lr.sigmoid(X @ k_theta.T)


# In[9]:


np.set_printoptions(suppress=True)
prob_matrix


# In[10]:


y_pred = np.argmax(prob_matrix, axis=1)


# In[11]:


y_answer = raw_y.copy()
y_answer[y_answer==10] = 0


# In[12]:


print(classification_report(y_answer, y_pred))


# In[ ]:




