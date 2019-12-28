#!/usr/bin/env python
# coding: utf-8

# In[23]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", style="white", palette=sns.color_palette("RdBu"))

import numpy as np
import pandas as pd
import scipy.io as sio

import sys
sys.path.append('..')

from helper import recommender as rcmd


# # # load data and setting up
# % Notes: X - num_movies (1682)  x num_features (10) matrix of movie features
# %        Theta - num_users (943)  x num_features (10) matrix of user features
# %        Y - num_movies x num_users matrix of user ratings of movies
# %        R - num_movies x num_users matrix, where R(i, j) = 1 if the
# %            i-th movie was rated by the j-th user
# # In[24]:


movies_mat = sio.loadmat('./data/ex8_movies.mat')
Y, R = movies_mat.get('Y'), movies_mat.get('R')

Y.shape, R.shape


# In[25]:


m, u = Y.shape
# m: how many movies
# u: how many users

n = 10  # how many features for a movie


# In[26]:


param_mat = sio.loadmat('./data/ex8_movieParams.mat')
theta, X = param_mat.get('Theta'), param_mat.get('X')

theta.shape, X.shape


# # cost
# <img style="float: left;" src="../img/rcmd_cost.png">

# In[27]:


# use subset of data to calculate the cost as in pdf...
users = 4
movies = 5
features = 3

X_sub = X[:movies, :features]
theta_sub = theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]

param_sub = rcmd.serialize(X_sub, theta_sub)

rcmd.cost(param_sub, Y_sub, R_sub, features)


# In[28]:


param = rcmd.serialize(X, theta)  # total real params

rcmd.cost(rcmd.serialize(X, theta), Y, R, 10)  # this is real total cost


# # gradient
# <img style="float: left;" src="../img/rcmd_gradient.png">

# In[29]:


n_movie, n_user = Y.shape

X_grad, theta_grad = rcmd.deserialize(rcmd.gradient(param, Y, R, 10),
                                      n_movie, n_user, 10)


# <img style="float: left;" src="../img/rcmd_vectorized_grad.png">

# In[30]:


assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape


# # regularized cost

# In[31]:


# in the ex8_confi.m, lambda = 1.5, and it's using sub data set
rcmd.regularized_cost(param_sub, Y_sub, R_sub, features, l=1.5)


# In[32]:


rcmd.regularized_cost(param, Y, R, 10, l=1)  # total regularized cost


# # regularized gradient

# <img style="float: left;" src="../img/rcmd_reg_grad.png">

# In[33]:


n_movie, n_user = Y.shape

X_grad, theta_grad = rcmd.deserialize(rcmd.regularized_gradient(param, Y, R, 10),
                                                                n_movie, n_user, 10)

assert X_grad.shape == X.shape
assert theta_grad.shape == theta.shape


# # parse `movie_id.txt`

# In[34]:


movie_list = []

with open('./data/movie_ids.txt', encoding='latin-1') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movie_list.append(' '.join(tokens[1:]))

movie_list = np.array(movie_list)


# # reproduce my ratings

# In[35]:


ratings = np.zeros(1682)

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5


# # prepare data

# In[36]:


Y, R = movies_mat.get('Y'), movies_mat.get('R')


Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
Y.shape


# In[37]:


R = np.insert(R, 0, ratings != 0, axis=1)
R.shape


# In[58]:


n_features = 50
n_movie, n_user = Y.shape
l = 10


# In[59]:


X = np.random.standard_normal((n_movie, n_features))
theta = np.random.standard_normal((n_user, n_features))

X.shape, theta.shape


# In[60]:


param = rcmd.serialize(X, theta)


# normalized ratings

# In[61]:


Y_norm = Y - Y.mean()
Y_norm.mean()


# # training

# In[62]:


import scipy.optimize as opt


# In[63]:


res = opt.minimize(fun=rcmd.regularized_cost,
                   x0=param,
                   args=(Y_norm, R, n_features, l),
                   method='TNC',
                   jac=rcmd.regularized_gradient)


# In[64]:


res


# In[65]:


X_trained, theta_trained = rcmd.deserialize(res.x, n_movie, n_user, n_features)
X_trained.shape, theta_trained.shape


# In[66]:


prediction = X_trained @ theta_trained.T


# In[67]:


my_preds = prediction[:, 0] + Y.mean()


# In[68]:


idx = np.argsort(my_preds)[::-1]  # Descending order
idx.shape


# In[69]:


# top ten idx
my_preds[idx][:10]


# In[70]:


for m in movie_list[idx][:10]:
    print(m)


# In[ ]:




