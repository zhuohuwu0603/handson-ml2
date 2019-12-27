#!/usr/bin/env python
# coding: utf-8

# # try different $\lambda$ constant

# In[5]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.append('..')

# load my own module
from helper import logistic_regression as lr


# # draw decision boundary
# * we want to find all those x which $X\times \theta = 0$
# * instead of solving polynomial equation, just create a coridate x,y grid that is dense enough, and find all those $X\times \theta$ that is close enough to 0, then plot them
# * [zip is its own inverse!](http://stackoverflow.com/a/19343/3943702)

# In[6]:


lr.draw_boundary(power=6, l=1)


# In[7]:


lr.draw_boundary(power=6, l=0)  # no regularization, over fitting


# In[8]:


lr.draw_boundary(power=6, l=100)  # underfitting


# In[ ]:




