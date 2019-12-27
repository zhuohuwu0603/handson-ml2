#!/usr/bin/env python
# coding: utf-8

# # scikit-learn-random forest

# Credits: Forked from [PyCon 2015 Scikit-learn Tutorial](https://github.com/jakevdp/sklearn_pycon2015) by Jake VanderPlas

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn.linear_model import LinearRegression
from scipy import stats
import pylab as pl

seaborn.set()


# ## Random Forest Classifier

# Random forests are an example of an *ensemble learner* built on decision trees.
# For this reason we'll start by discussing decision trees themselves.
# 
# Decision trees are extremely intuitive ways to classify or label objects: you simply ask a series of questions designed to zero-in on the classification:

# In[2]:


import fig_code
fig_code.plot_example_decision_tree()


# The binary splitting makes this extremely efficient.
# As always, though, the trick is to *ask the right questions*.
# This is where the algorithmic process comes in: in training a decision tree classifier, the algorithm looks at the features and decides which questions (or "splits") contain the most information.
# 
# ### Creating a Decision Tree
# 
# Here's an example of a decision tree classifier in scikit-learn. We'll start by defining some two-dimensional labeled data:

# In[3]:


from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');


# In[4]:


# We have some convenience functions in the repository that help 
from fig_code import visualize_tree, plot_tree_interactive

# Now using IPython's ``interact`` (available in IPython 2.0+, and requires a live kernel) we can view the decision tree splits:
plot_tree_interactive(X, y);


# Notice that at each increase in depth, every node is split in two **except** those nodes which contain only a single class.
# The result is a very fast **non-parametric** classification, and can be extremely useful in practice.
# 
# **Question: Do you see any problems with this?**

# ### Decision Trees and over-fitting
# 
# One issue with decision trees is that it is very easy to create trees which **over-fit** the data. That is, they are flexible enough that they can learn the structure of the noise in the data rather than the signal! For example, take a look at two trees built on two subsets of this dataset:

# In[5]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

plt.figure()
visualize_tree(clf, X[:200], y[:200], boundaries=False)
plt.figure()
visualize_tree(clf, X[-200:], y[-200:], boundaries=False)


# The details of the classifications are completely different! That is an indication of **over-fitting**: when you predict the value for a new point, the result is more reflective of the noise in the model rather than the signal.

# ## Ensembles of Estimators: Random Forests
# 
# One possible way to address over-fitting is to use an **Ensemble Method**: this is a meta-estimator which essentially averages the results of many individual estimators which over-fit the data. Somewhat surprisingly, the resulting estimates are much more robust and accurate than the individual estimates which make them up!
# 
# One of the most common ensemble methods is the **Random Forest**, in which the ensemble is made up of many decision trees which are in some way perturbed.
# 
# There are volumes of theory and precedent about how to randomize these trees, but as an example, let's imagine an ensemble of estimators fit on subsets of the data. We can get an idea of what these might look like as follows:

# In[6]:


def fit_randomized_tree(random_state=0):
    X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=2.0)
    clf = DecisionTreeClassifier(max_depth=15)
    
    rng = np.random.RandomState(random_state)
    i = np.arange(len(y))
    rng.shuffle(i)
    visualize_tree(clf, X[i[:250]], y[i[:250]], boundaries=False,
                   xlim=(X[:, 0].min(), X[:, 0].max()),
                   ylim=(X[:, 1].min(), X[:, 1].max()))
    
from IPython.html.widgets import interact
interact(fit_randomized_tree, random_state=[0, 100]);


# See how the details of the model change as a function of the sample, while the larger characteristics remain the same!
# The random forest classifier will do something similar to this, but use a combined version of all these trees to arrive at a final answer:

# In[7]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
visualize_tree(clf, X, y, boundaries=False);


# By averaging over 100 randomly perturbed models, we end up with an overall model which is a much better fit to our data!
# 
# *(Note: above we randomized the model through sub-sampling... Random Forests use more sophisticated means of randomization, which you can read about in, e.g. the [scikit-learn documentation](http://scikit-learn.org/stable/modules/ensemble.html#forest)*)

# Not good for random forest:
# lots of 0, few 1
# structured data like images, neural network might be better
# small data, might overfit
# high dimensional data, linear model might work better

# ## Random Forest Regressor

# Above we were considering random forests within the context of classification.
# Random forests can also be made to work in the case of regression (that is, continuous rather than categorical variables). The estimator to use for this is ``sklearn.ensemble.RandomForestRegressor``.
# 
# Let's quickly demonstrate how this can be used:

# In[8]:


from sklearn.ensemble import RandomForestRegressor

x = 10 * np.random.rand(100)

def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * np.random.randn(len(x))

    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt='o');


# In[9]:


xfit = np.linspace(0, 10, 1000)
yfit = RandomForestRegressor(100).fit(x[:, None], y).predict(xfit[:, None])
ytrue = model(xfit, 0)

plt.errorbar(x, y, 0.3, fmt='o')
plt.plot(xfit, yfit, '-r');
plt.plot(xfit, ytrue, '-k', alpha=0.5);


# As you can see, the non-parametric random forest model is flexible enough to fit the multi-period data, without us even specifying a multi-period model!
# 
# Tradeoff between simplicity and thinking about what your data is.
# 
# Feature engineering is important, need to know your domain: Fourier transform frequency distribution.

# ## Random Forest Limitations
# 
# The following data scenarios are not well suited for random forests:
# * y: lots of 0, few 1
# * Structured data like images where a neural network might be better
# * Small data size which might lead to overfitting
# * High dimensional data where a linear model might work better
