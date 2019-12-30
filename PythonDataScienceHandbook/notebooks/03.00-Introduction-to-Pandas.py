#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb) | [Contents](Index.ipynb) | [Introducing Pandas Objects](03.01-Introducing-Pandas-Objects.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.00-Introduction-to-Pandas.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # Data Manipulation with Pandas

# In the previous chapter, we dove into detail on NumPy and its ``ndarray`` object, which provides efficient storage and manipulation of dense typed arrays in Python.
# Here we'll build on this knowledge by looking in detail at the data structures provided by the Pandas library.
# Pandas is a newer package built on top of NumPy, and provides an efficient implementation of a ``DataFrame``.
# ``DataFrame``s are essentially multidimensional arrays with attached row and column labels, and often with heterogeneous types and/or missing data.
# As well as offering a convenient storage interface for labeled data, Pandas implements a number of powerful data operations familiar to users of both database frameworks and spreadsheet programs.
# 
# As we saw, NumPy's ``ndarray`` data structure provides essential features for the type of clean, well-organized data typically seen in numerical computing tasks.
# While it serves this purpose very well, its limitations become clear when we need more flexibility (e.g., attaching labels to data, working with missing data, etc.) and when attempting operations that do not map well to element-wise broadcasting (e.g., groupings, pivots, etc.), each of which is an important piece of analyzing the less structured data available in many forms in the world around us.
# Pandas, and in particular its ``Series`` and ``DataFrame`` objects, builds on the NumPy array structure and provides efficient access to these sorts of "data munging" tasks that occupy much of a data scientist's time.
# 
# In this chapter, we will focus on the mechanics of using ``Series``, ``DataFrame``, and related structures effectively.
# We will use examples drawn from real datasets where appropriate, but these examples are not necessarily the focus.

# ## Installing and Using Pandas
# 
# Installation of Pandas on your system requires NumPy to be installed, and if building the library from source, requires the appropriate tools to compile the C and Cython sources on which Pandas is built.
# Details on this installation can be found in the [Pandas documentation](http://pandas.pydata.org/).
# If you followed the advice outlined in the [Preface](00.00-Preface.ipynb) and used the Anaconda stack, you already have Pandas installed.
# 
# Once Pandas is installed, you can import it and check the version:

# In[1]:


import pandas
pandas.__version__


# Just as we generally import NumPy under the alias ``np``, we will import Pandas under the alias ``pd``:

# In[2]:


import pandas as pd


# This import convention will be used throughout the remainder of this book.

# ## Reminder about Built-In Documentation
# 
# As you read through this chapter, don't forget that IPython gives you the ability to quickly explore the contents of a package (by using the tab-completion feature) as well as the documentation of various functions (using the ``?`` character). (Refer back to [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) if you need a refresher on this.)
# 
# For example, to display all the contents of the pandas namespace, you can type
# 
# ```ipython
# In [3]: pd.<TAB>
# ```
# 
# And to display Pandas's built-in documentation, you can use this:
# 
# ```ipython
# In [4]: pd?
# ```
# 
# More detailed documentation, along with tutorials and other resources, can be found at http://pandas.pydata.org/.

# <!--NAVIGATION-->
# < [Structured Data: NumPy's Structured Arrays](02.09-Structured-Data-NumPy.ipynb) | [Contents](Index.ipynb) | [Introducing Pandas Objects](03.01-Introducing-Pandas-Objects.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.00-Introduction-to-Pandas.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
