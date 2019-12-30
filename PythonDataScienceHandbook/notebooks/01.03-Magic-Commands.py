#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) | [Contents](Index.ipynb) | [Input and Output History](01.04-Input-Output-History.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.03-Magic-Commands.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # IPython Magic Commands

# The previous two sections showed how IPython lets you use and explore Python efficiently and interactively.
# Here we'll begin discussing some of the enhancements that IPython adds on top of the normal Python syntax.
# These are known in IPython as *magic commands*, and are prefixed by the ``%`` character.
# These magic commands are designed to succinctly solve various common problems in standard data analysis.
# Magic commands come in two flavors: *line magics*, which are denoted by a single ``%`` prefix and operate on a single line of input, and *cell magics*, which are denoted by a double ``%%`` prefix and operate on multiple lines of input.
# We'll demonstrate and discuss a few brief examples here, and come back to more focused discussion of several useful magic commands later in the chapter.

# ## Pasting Code Blocks: ``%paste`` and ``%cpaste``
# 
# When working in the IPython interpreter, one common gotcha is that pasting multi-line code blocks can lead to unexpected errors, especially when indentation and interpreter markers are involved.
# A common case is that you find some example code on a website and want to paste it into your interpreter.
# Consider the following simple function:
# 
# ``` python
# >>> def donothing(x):
# ...     return x
# 
# ```
# The code is formatted as it would appear in the Python interpreter, and if you copy and paste this directly into IPython you get an error:
# 
# ```ipython
# In [2]: >>> def donothing(x):
#    ...:     ...     return x
#    ...:     
#   File "<ipython-input-20-5a66c8964687>", line 2
#     ...     return x
#                  ^
# SyntaxError: invalid syntax
# ```
# 
# In the direct paste, the interpreter is confused by the additional prompt characters.
# But never fear–IPython's ``%paste`` magic function is designed to handle this exact type of multi-line, marked-up input:
# 
# ```ipython
# In [3]: %paste
# >>> def donothing(x):
# ...     return x
# 
# ## -- End pasted text --
# ```
# 
# The ``%paste`` command both enters and executes the code, so now the function is ready to be used:
# 
# ```ipython
# In [4]: donothing(10)
# Out[4]: 10
# ```
# 
# A command with a similar intent is ``%cpaste``, which opens up an interactive multiline prompt in which you can paste one or more chunks of code to be executed in a batch:
# 
# ```ipython
# In [5]: %cpaste
# Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
# :>>> def donothing(x):
# :...     return x
# :--
# ```
# 
# These magic commands, like others we'll see, make available functionality that would be difficult or impossible in a standard Python interpreter.

# ## Running External Code: ``%run``
# As you begin developing more extensive code, you will likely find yourself working in both IPython for interactive exploration, as well as a text editor to store code that you want to reuse.
# Rather than running this code in a new window, it can be convenient to run it within your IPython session.
# This can be done with the ``%run`` magic.
# 
# For example, imagine you've created a ``myscript.py`` file with the following contents:
# 
# ```python
# #-------------------------------------
# # file: myscript.py
# 
# def square(x):
#     """square a number"""
#     return x ** 2
# 
# for N in range(1, 4):
#     print(N, "squared is", square(N))
# ```
# 
# You can execute this from your IPython session as follows:
# 
# ```ipython
# In [6]: %run myscript.py
# 1 squared is 1
# 2 squared is 4
# 3 squared is 9
# ```
# 
# Note also that after you've run this script, any functions defined within it are available for use in your IPython session:
# 
# ```ipython
# In [7]: square(5)
# Out[7]: 25
# ```
# 
# There are several options to fine-tune how your code is run; you can see the documentation in the normal way, by typing **``%run?``** in the IPython interpreter.

# ## Timing Code Execution: ``%timeit``
# Another example of a useful magic function is ``%timeit``, which will automatically determine the execution time of the single-line Python statement that follows it.
# For example, we may want to check the performance of a list comprehension:
# 
# ```ipython
# In [8]: %timeit L = [n ** 2 for n in range(1000)]
# 1000 loops, best of 3: 325 µs per loop
# ```
# 
# The benefit of ``%timeit`` is that for short commands it will automatically perform multiple runs in order to attain more robust results.
# For multi line statements, adding a second ``%`` sign will turn this into a cell magic that can handle multiple lines of input.
# For example, here's the equivalent construction with a ``for``-loop:
# 
# ```ipython
# In [9]: %%timeit
#    ...: L = []
#    ...: for n in range(1000):
#    ...:     L.append(n ** 2)
#    ...: 
# 1000 loops, best of 3: 373 µs per loop
# ```
# 
# We can immediately see that list comprehensions are about 10% faster than the equivalent ``for``-loop construction in this case.
# We'll explore ``%timeit`` and other approaches to timing and profiling code in [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb).

# ## Help on Magic Functions: ``?``, ``%magic``, and ``%lsmagic``
# 
# Like normal Python functions, IPython magic functions have docstrings, and this useful
# documentation can be accessed in the standard manner.
# So, for example, to read the documentation of the ``%timeit`` magic simply type this:
# 
# ```ipython
# In [10]: %timeit?
# ```
# 
# Documentation for other functions can be accessed similarly.
# To access a general description of available magic functions, including some examples, you can type this:
# 
# ```ipython
# In [11]: %magic
# ```
# 
# For a quick and simple list of all available magic functions, type this:
# 
# ```ipython
# In [12]: %lsmagic
# ```
# 
# Finally, I'll mention that it is quite straightforward to define your own magic functions if you wish.
# We won't discuss it here, but if you are interested, see the references listed in [More IPython Resources](01.08-More-IPython-Resources.ipynb).

# <!--NAVIGATION-->
# < [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) | [Contents](Index.ipynb) | [Input and Output History](01.04-Input-Output-History.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.03-Magic-Commands.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
