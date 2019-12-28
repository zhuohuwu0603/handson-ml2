#!/usr/bin/env python
# coding: utf-8

# Effect Size
# ======================
# 
# Credits: Forked from [CompStats](https://github.com/AllenDowney/CompStats) by Allen Downey.  License: [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/).

# In[1]:


from __future__ import print_function, division

import numpy
import scipy.stats

import matplotlib.pyplot as pyplot

from IPython.html.widgets import interact, fixed
from IPython.html import widgets

# seed the random number generator so we all get the same results
numpy.random.seed(17)

# some nice colors from http://colorbrewer2.org/
COLOR1 = '#7fc97f'
COLOR2 = '#beaed4'
COLOR3 = '#fdc086'
COLOR4 = '#ffff99'
COLOR5 = '#386cb0'

# get_ipython().run_line_magic('matplotlib', 'inline')


# To explore statistics that quantify effect size, we'll look at the difference in height between men and women.  I used data from the Behavioral Risk Factor Surveillance System (BRFSS) to estimate the mean and standard deviation of height in cm for adult women and men in the U.S.
# 
# I'll use `scipy.stats.norm` to represent the distributions.  The result is an `rv` object (which stands for random variable).

# In[2]:


mu1, sig1 = 178, 7.7
male_height = scipy.stats.norm(mu1, sig1)


# In[3]:


mu2, sig2 = 163, 7.3
female_height = scipy.stats.norm(mu2, sig2)


# The following function evaluates the normal (Gaussian) probability density function (PDF) within 4 standard deviations of the mean.  It takes and rv object and returns a pair of NumPy arrays.

# In[4]:


def eval_pdf(rv, num=4):
    mean, std = rv.mean(), rv.std()
    xs = numpy.linspace(mean - num*std, mean + num*std, 100)
    ys = rv.pdf(xs)
    return xs, ys


# Here's what the two distributions look like.

# In[5]:


xs, ys = eval_pdf(male_height)
pyplot.plot(xs, ys, label='male', linewidth=4, color=COLOR2)

xs, ys = eval_pdf(female_height)
pyplot.plot(xs, ys, label='female', linewidth=4, color=COLOR3)
pyplot.xlabel('height (cm)')
None


# Let's assume for now that those are the true distributions for the population.  Of course, in real life we never observe the true population distribution.  We generally have to work with a random sample.
# 
# I'll use `rvs` to generate random samples from the population distributions.  Note that these are totally random, totally representative samples, with no measurement error!

# In[6]:


male_sample = male_height.rvs(1000)


# In[7]:


female_sample = female_height.rvs(1000)


# Both samples are NumPy arrays.  Now we can compute sample statistics like the mean and standard deviation.

# In[8]:


mean1, std1 = male_sample.mean(), male_sample.std()
mean1, std1


# The sample mean is close to the population mean, but not exact, as expected.

# In[9]:


mean2, std2 = female_sample.mean(), female_sample.std()
mean2, std2


# And the results are similar for the female sample.
# 
# Now, there are many ways to describe the magnitude of the difference between these distributions.  An obvious one is the difference in the means:

# In[10]:


difference_in_means = male_sample.mean() - female_sample.mean()
difference_in_means # in cm


# On average, men are 14--15 centimeters taller.  For some applications, that would be a good way to describe the difference, but there are a few problems:
# 
# * Without knowing more about the distributions (like the standard deviations) it's hard to interpret whether a difference like 15 cm is a lot or not.
# 
# * The magnitude of the difference depends on the units of measure, making it hard to compare across different studies.
# 
# There are a number of ways to quantify the difference between distributions.  A simple option is to express the difference as a percentage of the mean.
# 

# In[11]:


# Exercise: what is the relative difference in means, expressed as a percentage?

relative_difference = difference_in_means / male_sample.mean()
relative_difference * 100   # percent


# But a problem with relative differences is that you have to choose which mean to express them relative to.

# In[12]:


relative_difference = difference_in_means / female_sample.mean()
relative_difference * 100    # percent


# Part Two
# ========
# 
# An alternative way to express the difference between distributions is to see how much they overlap.  To define overlap, we choose a threshold between the two means.  The simple threshold is the midpoint between the means:

# In[13]:


simple_thresh = (mean1 + mean2) / 2
simple_thresh


# A better, but slightly more complicated threshold is the place where the PDFs cross.

# In[14]:


thresh = (std1 * mean2 + std2 * mean1) / (std1 + std2)
thresh


# In this example, there's not much difference between the two thresholds.
# 
# Now we can count how many men are below the threshold:

# In[15]:


male_below_thresh = sum(male_sample < thresh)
male_below_thresh


# And how many women are above it:

# In[16]:


female_above_thresh = sum(female_sample > thresh)
female_above_thresh


# The "overlap" is the total area under the curves that ends up on the wrong side of the threshold.

# In[17]:


overlap = male_below_thresh / len(male_sample) + female_above_thresh / len(female_sample)
overlap


# Or in more practical terms, you might report the fraction of people who would be misclassified if you tried to use height to guess sex:

# In[18]:


misclassification_rate = overlap / 2
misclassification_rate


# Another way to quantify the difference between distributions is what's called "probability of superiority", which is a problematic term, but in this context it's the probability that a randomly-chosen man is taller than a randomly-chosen woman.

# In[19]:


# Exercise: suppose I choose a man and a woman at random.
# What is the probability that the man is taller?
sum(x > y for x, y in zip(male_sample, female_sample)) / len(male_sample)


# Overlap (or misclassification rate) and "probability of superiority" have two good properties:
# 
# * As probabilities, they don't depend on units of measure, so they are comparable between studies.
# 
# * They are expressed in operational terms, so a reader has a sense of what practical effect the difference makes.
# 
# There is one other common way to express the difference between distributions.  Cohen's $d$ is the difference in means, standardized by dividing by the standard deviation.  Here's a function that computes it:
# 

# In[20]:


def CohenEffectSize(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / numpy.sqrt(pooled_var)
    return d


# Computing the denominator is a little complicated; in fact, people have proposed several ways to do it.  This implementation uses the "pooled standard deviation", which is a weighted average of the standard deviations of the two groups.
# 
# And here's the result for the difference in height between men and women.

# In[21]:


CohenEffectSize(male_sample, female_sample)


# Most people don't have a good sense of how big $d=1.9$ is, so let's make a visualization to get calibrated.
# 
# Here's a function that encapsulates the code we already saw for computing overlap and probability of superiority.

# In[22]:


def overlap_superiority(control, treatment, n=1000):
    """Estimates overlap and superiority based on a sample.
    
    control: scipy.stats rv object
    treatment: scipy.stats rv object
    n: sample size
    """
    control_sample = control.rvs(n)
    treatment_sample = treatment.rvs(n)
    thresh = (control.mean() + treatment.mean()) / 2
    
    control_above = sum(control_sample > thresh)
    treatment_below = sum(treatment_sample < thresh)
    overlap = (control_above + treatment_below) / n
    
    superiority = sum(x > y for x, y in zip(treatment_sample, control_sample)) / n
    return overlap, superiority


# Here's the function that takes Cohen's $d$, plots normal distributions with the given effect size, and prints their overlap and superiority.

# In[23]:


def plot_pdfs(cohen_d=2):
    """Plot PDFs for distributions that differ by some number of stds.
    
    cohen_d: number of standard deviations between the means
    """
    control = scipy.stats.norm(0, 1)
    treatment = scipy.stats.norm(cohen_d, 1)
    xs, ys = eval_pdf(control)
    pyplot.fill_between(xs, ys, label='control', color=COLOR3, alpha=0.7)

    xs, ys = eval_pdf(treatment)
    pyplot.fill_between(xs, ys, label='treatment', color=COLOR2, alpha=0.7)
    
    o, s = overlap_superiority(control, treatment)
    print('overlap', o)
    print('superiority', s)


# Here's an example that demonstrates the function:

# In[24]:


plot_pdfs(2)


# And an interactive widget you can use to visualize what different values of $d$ mean:

# In[25]:


slider = widgets.FloatSliderWidget(min=0, max=4, value=2)
interact(plot_pdfs, cohen_d=slider)
None


# Cohen's $d$ has a few nice properties:
# 
# * Because mean and standard deviation have the same units, their ratio is dimensionless, so we can compare $d$ across different studies.
# 
# * In fields that commonly use $d$, people are calibrated to know what values should be considered big, surprising, or important.
# 
# * Given $d$ (and the assumption that the distributions are normal), you can compute overlap, superiority, and related statistics.

# In summary, the best way to report effect size often depends on the audience and your goals.  There is often a tradeoff between summary statistics that have good technical properties and statistics that are meaningful to a general audience.
