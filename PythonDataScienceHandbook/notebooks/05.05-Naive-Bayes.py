#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Feature Engineering](05.04-Feature-Engineering.ipynb) | [Contents](Index.ipynb) | [In Depth: Linear Regression](05.06-Linear-Regression.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # In Depth: Naive Bayes Classification

# The previous four sections have given a general overview of the concepts of machine learning.
# In this section and the ones that follow, we will be taking a closer look at several specific algorithms for supervised and unsupervised learning, starting here with naive Bayes classification.
# 
# Naive Bayes models are a group of extremely fast and simple classification algorithms that are often suitable for very high-dimensional datasets.
# Because they are so fast and have so few tunable parameters, they end up being very useful as a quick-and-dirty baseline for a classification problem.
# This section will focus on an intuitive explanation of how naive Bayes classifiers work, followed by a couple examples of them in action on some datasets.

# ## Bayesian Classification
# 
# Naive Bayes classifiers are built on Bayesian classification methods.
# These rely on Bayes's theorem, which is an equation describing the relationship of conditional probabilities of statistical quantities.
# In Bayesian classification, we're interested in finding the probability of a label given some observed features, which we can write as $P(L~|~{\rm features})$.
# Bayes's theorem tells us how to express this in terms of quantities we can compute more directly:
# 
# $$
# P(L~|~{\rm features}) = \frac{P({\rm features}~|~L)P(L)}{P({\rm features})}
# $$
# 
# If we are trying to decide between two labels—let's call them $L_1$ and $L_2$—then one way to make this decision is to compute the ratio of the posterior probabilities for each label:
# 
# $$
# \frac{P(L_1~|~{\rm features})}{P(L_2~|~{\rm features})} = \frac{P({\rm features}~|~L_1)}{P({\rm features}~|~L_2)}\frac{P(L_1)}{P(L_2)}
# $$
# 
# All we need now is some model by which we can compute $P({\rm features}~|~L_i)$ for each label.
# Such a model is called a *generative model* because it specifies the hypothetical random process that generates the data.
# Specifying this generative model for each label is the main piece of the training of such a Bayesian classifier.
# The general version of such a training step is a very difficult task, but we can make it simpler through the use of some simplifying assumptions about the form of this model.
# 
# This is where the "naive" in "naive Bayes" comes in: if we make very naive assumptions about the generative model for each label, we can find a rough approximation of the generative model for each class, and then proceed with the Bayesian classification.
# Different types of naive Bayes classifiers rest on different naive assumptions about the data, and we will examine a few of these in the following sections.
# 
# We begin with the standard imports:

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# ## Gaussian Naive Bayes
# 
# Perhaps the easiest naive Bayes classifier to understand is Gaussian naive Bayes.
# In this classifier, the assumption is that *data from each label is drawn from a simple Gaussian distribution*.
# Imagine that you have the following data:

# In[2]:


from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');


# One extremely fast way to create a simple model is to assume that the data is described by a Gaussian distribution with no covariance between dimensions.
# This model can be fit by simply finding the mean and standard deviation of the points within each label, which is all you need to define such a distribution.
# The result of this naive Gaussian assumption is shown in the following figure:

# ![(run code in Appendix to generate image)](figures/05.05-gaussian-NB.png)
# [figure source in Appendix](06.00-Figure-Code.ipynb#Gaussian-Naive-Bayes)

# The ellipses here represent the Gaussian generative model for each label, with larger probability toward the center of the ellipses.
# With this generative model in place for each class, we have a simple recipe to compute the likelihood $P({\rm features}~|~L_1)$ for any data point, and thus we can quickly compute the posterior ratio and determine which label is the most probable for a given point.
# 
# This procedure is implemented in Scikit-Learn's ``sklearn.naive_bayes.GaussianNB`` estimator:

# In[3]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);


# Now let's generate some new data and predict the label:

# In[4]:


rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)


# Now we can plot this new data to get an idea of where the decision boundary is:

# In[5]:


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);


# We see a slightly curved boundary in the classifications—in general, the boundary in Gaussian naive Bayes is quadratic.
# 
# A nice piece of this Bayesian formalism is that it naturally allows for probabilistic classification, which we can compute using the ``predict_proba`` method:

# In[6]:


yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)


# The columns give the posterior probabilities of the first and second label, respectively.
# If you are looking for estimates of uncertainty in your classification, Bayesian approaches like this can be a useful approach.
# 
# Of course, the final classification will only be as good as the model assumptions that lead to it, which is why Gaussian naive Bayes often does not produce very good results.
# Still, in many cases—especially as the number of features becomes large—this assumption is not detrimental enough to prevent Gaussian naive Bayes from being a useful method.

# ## Multinomial Naive Bayes
# 
# The Gaussian assumption just described is by no means the only simple assumption that could be used to specify the generative distribution for each label.
# Another useful example is multinomial naive Bayes, where the features are assumed to be generated from a simple multinomial distribution.
# The multinomial distribution describes the probability of observing counts among a number of categories, and thus multinomial naive Bayes is most appropriate for features that represent counts or count rates.
# 
# The idea is precisely the same as before, except that instead of modeling the data distribution with the best-fit Gaussian, we model the data distribuiton with a best-fit multinomial distribution.

# ### Example: Classifying Text
# 
# One place where multinomial naive Bayes is often used is in text classification, where the features are related to word counts or frequencies within the documents to be classified.
# We discussed the extraction of such features from text in [Feature Engineering](05.04-Feature-Engineering.ipynb); here we will use the sparse word count features from the 20 Newsgroups corpus to show how we might classify these short documents into categories.
# 
# Let's download the data and take a look at the target names:

# In[7]:


from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names


# For simplicity here, we will select just a few of these categories, and download the training and testing set:

# In[8]:


categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)


# Here is a representative entry from the data:

# In[9]:


print(train.data[5])


# In order to use this data for machine learning, we need to be able to convert the content of each string into a vector of numbers.
# For this we will use the TF-IDF vectorizer (discussed in [Feature Engineering](05.04-Feature-Engineering.ipynb)), and create a pipeline that attaches it to a multinomial naive Bayes classifier:

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())


# With this pipeline, we can apply the model to the training data, and predict labels for the test data:

# In[11]:


model.fit(train.data, train.target)
labels = model.predict(test.data)


# Now that we have predicted the labels for the test data, we can evaluate them to learn about the performance of the estimator.
# For example, here is the confusion matrix between the true and predicted labels for the test data:

# In[12]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');


# Evidently, even this very simple classifier can successfully separate space talk from computer talk, but it gets confused between talk about religion and talk about Christianity.
# This is perhaps an expected area of confusion!
# 
# The very cool thing here is that we now have the tools to determine the category for *any* string, using the ``predict()`` method of this pipeline.
# Here's a quick utility function that will return the prediction for a single string:

# In[13]:


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# Let's try it out:

# In[14]:


predict_category('sending a payload to the ISS')


# In[15]:


predict_category('discussing islam vs atheism')


# In[16]:


predict_category('determining the screen resolution')


# Remember that this is nothing more sophisticated than a simple probability model for the (weighted) frequency of each word in the string; nevertheless, the result is striking.
# Even a very naive algorithm, when used carefully and trained on a large set of high-dimensional data, can be surprisingly effective.

# ## When to Use Naive Bayes
# 
# Because naive Bayesian classifiers make such stringent assumptions about data, they will generally not perform as well as a more complicated model.
# That said, they have several advantages:
# 
# - They are extremely fast for both training and prediction
# - They provide straightforward probabilistic prediction
# - They are often very easily interpretable
# - They have very few (if any) tunable parameters
# 
# These advantages mean a naive Bayesian classifier is often a good choice as an initial baseline classification.
# If it performs suitably, then congratulations: you have a very fast, very interpretable classifier for your problem.
# If it does not perform well, then you can begin exploring more sophisticated models, with some baseline knowledge of how well they should perform.
# 
# Naive Bayes classifiers tend to perform especially well in one of the following situations:
# 
# - When the naive assumptions actually match the data (very rare in practice)
# - For very well-separated categories, when model complexity is less important
# - For very high-dimensional data, when model complexity is less important
# 
# The last two points seem distinct, but they actually are related: as the dimension of a dataset grows, it is much less likely for any two points to be found close together (after all, they must be close in *every single dimension* to be close overall).
# This means that clusters in high dimensions tend to be more separated, on average, than clusters in low dimensions, assuming the new dimensions actually add information.
# For this reason, simplistic classifiers like naive Bayes tend to work as well or better than more complicated classifiers as the dimensionality grows: once you have enough data, even a simple model can be very powerful.

# <!--NAVIGATION-->
# < [Feature Engineering](05.04-Feature-Engineering.ipynb) | [Contents](Index.ipynb) | [In Depth: Linear Regression](05.06-Linear-Regression.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.05-Naive-Bayes.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
