#!/usr/bin/env python
# coding: utf-8

# # notes
# * Now I know I should think in column vector, and Tensorflow is very picky about the shape of data. But in numpy, the normal 1D ndarray is represented as column vector already. If I reshape $\mathbb{R}^n$ as $\mathbb{R}^{n\times1}$, It's not the same as column vector anymore. It's Matrix with 1 column. And I got troubles with scipy optimizer. 
# * So I should just treat tensorflow's data as special case. Keep using the convention of numpy world.

# In[1]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import sys
sys.path.append('..')

from helper import logistic_regression as lr  # my own module
from helper import general as general

from sklearn.metrics import classification_report


# In[2]:


# prepare data
data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
data.head()


# In[3]:


X = general.get_X(data)
print(X.shape)

y = general.get_y(data)
print(y.shape)


# # sigmoid function

# In[4]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01),
        lr.sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1,1.1))
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)


# # cost function
# > * $max(\ell(\theta)) = min(-\ell(\theta))$  
# > * choose $-\ell(\theta)$ as the cost function
# 
# <img style="float: left;" src="../img/logistic_cost.png">

# In[5]:


theta = theta=np.zeros(3) # X(m*n) so theta is n*1
theta


# In[6]:


lr.cost(theta, X, y)


# looking good, be careful of the data shape

# # gradient
# * this is batch gradient  
# * translate this into vector computation $\frac{1}{m} X^T( Sigmoid(X\theta) - y )$
# 
# <img style="float: left;" src="../img/logistic_gradient.png">

# In[7]:


lr.gradient(theta, X, y)


# # fit the parameter
# > * here I'm using [`scipy.optimize.minimize`](http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) to find the parameters  
# > * and I use this model without understanding.... what is `Jacobian` ...

# In[8]:


import scipy.optimize as opt


# In[9]:


res = opt.minimize(fun=lr.cost, x0=theta, args=(X, y), method='Newton-CG', jac=lr.gradient)


# In[10]:


print(res)


# # predict and validate from training set
# > now we are using training set to evaluate the model, this is not the best practice, but the course just begin, I guess Andrew will cover how to do model validation properlly later

# In[11]:


final_theta = res.x
y_pred = lr.predict(X, final_theta)

print(classification_report(y, y_pred))


# # find the decision boundary
# http://stats.stackexchange.com/questions/93569/why-is-logistic-regression-a-linear-classifier
# > $X \times \theta = 0$  (this is the line)

# In[12]:


print(res.x) # this is final theta


# In[13]:


coef = -(res.x / res.x[2])  # find the equation
print(coef)

x = np.arange(130, step=0.1)
y = coef[0] + coef[1]*x


# In[14]:


data.describe()  # find the range of x and y


# > you know the intercept would be around 125 for both x and y

# In[15]:


sns.set(context="notebook", style="ticks", font_scale=1.5)

sns.lmplot('exam1', 'exam2', hue='admitted', data=data, 
           size=6, 
           fit_reg=False, 
           scatter_kws={"s": 25}
          )

plt.plot(x, y, 'grey')
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.title('Decision Boundary')


# In[ ]:




