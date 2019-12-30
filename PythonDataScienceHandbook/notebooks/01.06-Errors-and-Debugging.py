#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) | [Contents](Index.ipynb) | [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.06-Errors-and-Debugging.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # Errors and Debugging

# Code development and data analysis always require a bit of trial and error, and IPython contains tools to streamline this process.
# This section will briefly cover some options for controlling Python's exception reporting, followed by exploring tools for debugging errors in code.

# ## Controlling Exceptions: ``%xmode``
# 
# Most of the time when a Python script fails, it will raise an Exception.
# When the interpreter hits one of these exceptions, information about the cause of the error can be found in the *traceback*, which can be accessed from within Python.
# With the ``%xmode`` magic function, IPython allows you to control the amount of information printed when the exception is raised.
# Consider the following code:

# In[1]:


def func1(a, b):
    return a / b

def func2(x):
    a = x
    b = x - 1
    return func1(a, b)


# In[2]:


func2(1)


# Calling ``func2`` results in an error, and reading the printed trace lets us see exactly what happened.
# By default, this trace includes several lines showing the context of each step that led to the error.
# Using the ``%xmode`` magic function (short for *Exception mode*), we can change what information is printed.
# 
# ``%xmode`` takes a single argument, the mode, and there are three possibilities: ``Plain``, ``Context``, and ``Verbose``.
# The default is ``Context``, and gives output like that just shown before.
# ``Plain`` is more compact and gives less information:

# In[3]:


get_ipython().run_line_magic('xmode', 'Plain')


# In[4]:


func2(1)


# The ``Verbose`` mode adds some extra information, including the arguments to any functions that are called:

# In[5]:


get_ipython().run_line_magic('xmode', 'Verbose')


# In[6]:


func2(1)


# This extra information can help narrow-in on why the exception is being raised.
# So why not use the ``Verbose`` mode all the time?
# As code gets complicated, this kind of traceback can get extremely long.
# Depending on the context, sometimes the brevity of ``Default`` mode is easier to work with.

# ## Debugging: When Reading Tracebacks Is Not Enough
# 
# The standard Python tool for interactive debugging is ``pdb``, the Python debugger.
# This debugger lets the user step through the code line by line in order to see what might be causing a more difficult error.
# The IPython-enhanced version of this is ``ipdb``, the IPython debugger.
# 
# There are many ways to launch and use both these debuggers; we won't cover them fully here.
# Refer to the online documentation of these two utilities to learn more.
# 
# In IPython, perhaps the most convenient interface to debugging is the ``%debug`` magic command.
# If you call it after hitting an exception, it will automatically open an interactive debugging prompt at the point of the exception.
# The ``ipdb`` prompt lets you explore the current state of the stack, explore the available variables, and even run Python commands!
# 
# Let's look at the most recent exception, then do some basic tasks–print the values of ``a`` and ``b``, and type ``quit`` to quit the debugging session:

# In[7]:


get_ipython().run_line_magic('debug', '')


# The interactive debugger allows much more than this, though–we can even step up and down through the stack and explore the values of variables there:

# In[8]:


get_ipython().run_line_magic('debug', '')


# This allows you to quickly find out not only what caused the error, but what function calls led up to the error.
# 
# If you'd like the debugger to launch automatically whenever an exception is raised, you can use the ``%pdb`` magic function to turn on this automatic behavior:

# In[9]:


get_ipython().run_line_magic('xmode', 'Plain')
get_ipython().run_line_magic('pdb', 'on')
func2(1)


# Finally, if you have a script that you'd like to run from the beginning in interactive mode, you can run it with the command ``%run -d``, and use the ``next`` command to step through the lines of code interactively.

# ### Partial list of debugging commands
# 
# There are many more available commands for interactive debugging than we've listed here; the following table contains a description of some of the more common and useful ones:
# 
# | Command         |  Description                                                |
# |-----------------|-------------------------------------------------------------|
# | ``list``        | Show the current location in the file                       |
# | ``h(elp)``      | Show a list of commands, or find help on a specific command |
# | ``q(uit)``      | Quit the debugger and the program                           |
# | ``c(ontinue)``  | Quit the debugger, continue in the program                  |
# | ``n(ext)``      | Go to the next step of the program                          |
# | ``<enter>``     | Repeat the previous command                                 |
# | ``p(rint)``     | Print variables                                             |
# | ``s(tep)``      | Step into a subroutine                                      |
# | ``r(eturn)``    | Return out of a subroutine                                  |
# 
# For more information, use the ``help`` command in the debugger, or take a look at ``ipdb``'s [online documentation](https://github.com/gotcha/ipdb).

# <!--NAVIGATION-->
# < [IPython and Shell Commands](01.05-IPython-And-Shell-Commands.ipynb) | [Contents](Index.ipynb) | [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.06-Errors-and-Debugging.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
