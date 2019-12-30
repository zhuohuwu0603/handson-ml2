#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Preface](00.00-Preface.ipynb) | [Contents](Index.ipynb) | [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.00-IPython-Beyond-Normal-Python.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # IPython: Beyond Normal Python

# There are many options for development environments for Python, and I'm often asked which one I use in my own work.
# My answer sometimes surprises people: my preferred environment is [IPython](http://ipython.org/) plus a text editor (in my case, Emacs or Atom depending on my mood).
# IPython (short for *Interactive Python*) was started in 2001 by Fernando Perez as an enhanced Python interpreter, and has since grown into a project aiming to provide, in Perez's words, "Tools for the entire life cycle of research computing."
# If Python is the engine of our data science task, you might think of IPython as the interactive control panel.
# 
# As well as being a useful interactive interface to Python, IPython also provides a number of useful syntactic additions to the language; we'll cover the most useful of these additions here.
# In addition, IPython is closely tied with the [Jupyter project](http://jupyter.org), which provides a browser-based notebook that is useful for development, collaboration, sharing, and even publication of data science results.
# The IPython notebook is actually a special case of the broader Jupyter notebook structure, which encompasses notebooks for Julia, R, and other programming languages.
# As an example of the usefulness of the notebook format, look no further than the page you are reading: the entire manuscript for this book was composed as a set of IPython notebooks.
# 
# IPython is about using Python effectively for interactive scientific and data-intensive computing.
# This chapter will start by stepping through some of the IPython features that are useful to the practice of data science, focusing especially on the syntax it offers beyond the standard features of Python.
# Next, we will go into a bit more depth on some of the more useful "magic commands" that can speed-up common tasks in creating and using data science code.
# Finally, we will touch on some of the features of the notebook that make it useful in understanding data and sharing results.

# ## Shell or Notebook?
# 
# There are two primary means of using IPython that we'll discuss in this chapter: the IPython shell and the IPython notebook.
# The bulk of the material in this chapter is relevant to both, and the examples will switch between them depending on what is most convenient.
# In the few sections that are relevant to just one or the other, we will explicitly state that fact.
# Before we start, some words on how to launch the IPython shell and IPython notebook.

# ### Launching the IPython Shell
# 
# This chapter, like most of this book, is not designed to be absorbed passively.
# I recommend that as you read through it, you follow along and experiment with the tools and syntax we cover: the muscle-memory you build through doing this will be far more useful than the simple act of reading about it.
# Start by launching the IPython interpreter by typing **``ipython``** on the command-line; alternatively, if you've installed a distribution like Anaconda or EPD, there may be a launcher specific to your system (we'll discuss this more fully in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)).
# 
# Once you do this, you should see a prompt like the following:
# ```
# IPython 4.0.1 -- An enhanced Interactive Python.
# ?         -> Introduction and overview of IPython's features.
# %quickref -> Quick reference.
# help      -> Python's own help system.
# object?   -> Details about 'object', use 'object??' for extra details.
# In [1]:
# ```
# With that, you're ready to follow along.

# ### Launching the Jupyter Notebook
# 
# The Jupyter notebook is a browser-based graphical interface to the IPython shell, and builds on it a rich set of dynamic display capabilities.
# As well as executing Python/IPython statements, the notebook allows the user to include formatted text, static and dynamic visualizations, mathematical equations, JavaScript widgets, and much more.
# Furthermore, these documents can be saved in a way that lets other people open them and execute the code on their own systems.
# 
# Though the IPython notebook is viewed and edited through your web browser window, it must connect to a running Python process in order to execute code.
# This process (known as a "kernel") can be started by running the following command in your system shell:
# 
# ```
# $ jupyter notebook
# ```
# 
# This command will launch a local web server that will be visible to your browser.
# It immediately spits out a log showing what it is doing; that log will look something like this:
# 
# ```
# $ jupyter notebook
# [NotebookApp] Serving notebooks from local directory: /Users/jakevdp/PythonDataScienceHandbook
# [NotebookApp] 0 active kernels 
# [NotebookApp] The IPython Notebook is running at: http://localhost:8888/
# [NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
# ```
# 
# Upon issuing the command, your default browser should automatically open and navigate to the listed local URL;
# the exact address will depend on your system.
# If the browser does not open automatically, you can open a window and manually open this address (*http://localhost:8888/* in this example).

# <!--NAVIGATION-->
# < [Preface](00.00-Preface.ipynb) | [Contents](Index.ipynb) | [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.00-IPython-Beyond-Normal-Python.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
