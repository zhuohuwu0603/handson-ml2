#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) | [Contents](Index.ipynb) | [More IPython Resources](01.08-More-IPython-Resources.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.07-Timing-and-Profiling.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # Profiling and Timing Code

# In the process of developing code and creating data processing pipelines, there are often trade-offs you can make between various implementations.
# Early in developing your algorithm, it can be counterproductive to worry about such things. As Donald Knuth famously quipped, "We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil."
# 
# But once you have your code working, it can be useful to dig into its efficiency a bit.
# Sometimes it's useful to check the execution time of a given command or set of commands; other times it's useful to dig into a multiline process and determine where the bottleneck lies in some complicated series of operations.
# IPython provides access to a wide array of functionality for this kind of timing and profiling of code.
# Here we'll discuss the following IPython magic commands:
# 
# - ``%time``: Time the execution of a single statement
# - ``%timeit``: Time repeated execution of a single statement for more accuracy
# - ``%prun``: Run code with the profiler
# - ``%lprun``: Run code with the line-by-line profiler
# - ``%memit``: Measure the memory use of a single statement
# - ``%mprun``: Run code with the line-by-line memory profiler
# 
# The last four commands are not bundled with IPython–you'll need to get the ``line_profiler`` and ``memory_profiler`` extensions, which we will discuss in the following sections.

# ## Timing Code Snippets: ``%timeit`` and ``%time``
# 
# We saw the ``%timeit`` line-magic and ``%%timeit`` cell-magic in the introduction to magic functions in [IPython Magic Commands](01.03-Magic-Commands.ipynb); it can be used to time the repeated execution of snippets of code:

# In[1]:


get_ipython().run_line_magic('timeit', 'sum(range(100))')


# Note that because this operation is so fast, ``%timeit`` automatically does a large number of repetitions.
# For slower commands, ``%timeit`` will automatically adjust and perform fewer repetitions:

# In[2]:


get_ipython().run_cell_magic('timeit', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')


# Sometimes repeating an operation is not the best option.
# For example, if we have a list that we'd like to sort, we might be misled by a repeated operation.
# Sorting a pre-sorted list is much faster than sorting an unsorted list, so the repetition will skew the result:

# In[3]:


import random
L = [random.random() for i in range(100000)]
get_ipython().run_line_magic('timeit', 'L.sort()')


# For this, the ``%time`` magic function may be a better choice. It also is a good choice for longer-running commands, when short, system-related delays are unlikely to affect the result.
# Let's time the sorting of an unsorted and a presorted list:

# In[4]:


import random
L = [random.random() for i in range(100000)]
print("sorting an unsorted list:")
get_ipython().run_line_magic('time', 'L.sort()')


# In[5]:


print("sorting an already sorted list:")
get_ipython().run_line_magic('time', 'L.sort()')


# Notice how much faster the presorted list is to sort, but notice also how much longer the timing takes with ``%time`` versus ``%timeit``, even for the presorted list!
# This is a result of the fact that ``%timeit`` does some clever things under the hood to prevent system calls from interfering with the timing.
# For example, it prevents cleanup of unused Python objects (known as *garbage collection*) which might otherwise affect the timing.
# For this reason, ``%timeit`` results are usually noticeably faster than ``%time`` results.
# 
# For ``%time`` as with ``%timeit``, using the double-percent-sign cell magic syntax allows timing of multiline scripts:

# In[6]:


get_ipython().run_cell_magic('time', '', 'total = 0\nfor i in range(1000):\n    for j in range(1000):\n        total += i * (-1) ** j')


# For more information on ``%time`` and ``%timeit``, as well as their available options, use the IPython help functionality (i.e., type ``%time?`` at the IPython prompt).

# ## Profiling Full Scripts: ``%prun``
# 
# A program is made of many single statements, and sometimes timing these statements in context is more important than timing them on their own.
# Python contains a built-in code profiler (which you can read about in the Python documentation), but IPython offers a much more convenient way to use this profiler, in the form of the magic function ``%prun``.
# 
# By way of example, we'll define a simple function that does some calculations:

# In[7]:


def sum_of_lists(N):
    total = 0
    for i in range(5):
        L = [j ^ (j >> i) for j in range(N)]
        total += sum(L)
    return total


# Now we can call ``%prun`` with a function call to see the profiled results:

# In[8]:


get_ipython().run_line_magic('prun', 'sum_of_lists(1000000)')


# In the notebook, the output is printed to the pager, and looks something like this:
# 
# ```
# 14 function calls in 0.714 seconds
# 
#    Ordered by: internal time
# 
#    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
#         5    0.599    0.120    0.599    0.120 <ipython-input-19>:4(<listcomp>)
#         5    0.064    0.013    0.064    0.013 {built-in method sum}
#         1    0.036    0.036    0.699    0.699 <ipython-input-19>:1(sum_of_lists)
#         1    0.014    0.014    0.714    0.714 <string>:1(<module>)
#         1    0.000    0.000    0.714    0.714 {built-in method exec}
# ```
# 
# The result is a table that indicates, in order of total time on each function call, where the execution is spending the most time. In this case, the bulk of execution time is in the list comprehension inside ``sum_of_lists``.
# From here, we could start thinking about what changes we might make to improve the performance in the algorithm.
# 
# For more information on ``%prun``, as well as its available options, use the IPython help functionality (i.e., type ``%prun?`` at the IPython prompt).

# ## Line-By-Line Profiling with ``%lprun``
# 
# The function-by-function profiling of ``%prun`` is useful, but sometimes it's more convenient to have a line-by-line profile report.
# This is not built into Python or IPython, but there is a ``line_profiler`` package available for installation that can do this.
# Start by using Python's packaging tool, ``pip``, to install the ``line_profiler`` package:
# 
# ```
# $ pip install line_profiler
# ```
# 
# Next, you can use IPython to load the ``line_profiler`` IPython extension, offered as part of this package:

# In[9]:


get_ipython().run_line_magic('load_ext', 'line_profiler')


# Now the ``%lprun`` command will do a line-by-line profiling of any function–in this case, we need to tell it explicitly which functions we're interested in profiling:

# In[10]:


get_ipython().run_line_magic('lprun', '-f sum_of_lists sum_of_lists(5000)')


# As before, the notebook sends the result to the pager, but it looks something like this:
# 
# ```
# Timer unit: 1e-06 s
# 
# Total time: 0.009382 s
# File: <ipython-input-19-fa2be176cc3e>
# Function: sum_of_lists at line 1
# 
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      1                                           def sum_of_lists(N):
#      2         1            2      2.0      0.0      total = 0
#      3         6            8      1.3      0.1      for i in range(5):
#      4         5         9001   1800.2     95.9          L = [j ^ (j >> i) for j in range(N)]
#      5         5          371     74.2      4.0          total += sum(L)
#      6         1            0      0.0      0.0      return total
# ```
# 
# The information at the top gives us the key to reading the results: the time is reported in microseconds and we can see where the program is spending the most time.
# At this point, we may be able to use this information to modify aspects of the script and make it perform better for our desired use case.
# 
# For more information on ``%lprun``, as well as its available options, use the IPython help functionality (i.e., type ``%lprun?`` at the IPython prompt).

# ## Profiling Memory Use: ``%memit`` and ``%mprun``
# 
# Another aspect of profiling is the amount of memory an operation uses.
# This can be evaluated with another IPython extension, the ``memory_profiler``.
# As with the ``line_profiler``, we start by ``pip``-installing the extension:
# 
# ```
# $ pip install memory_profiler
# ```
# 
# Then we can use IPython to load the extension:

# In[12]:


get_ipython().run_line_magic('load_ext', 'memory_profiler')


# The memory profiler extension contains two useful magic functions: the ``%memit`` magic (which offers a memory-measuring equivalent of ``%timeit``) and the ``%mprun`` function (which offers a memory-measuring equivalent of ``%lprun``).
# The ``%memit`` function can be used rather simply:

# In[13]:


get_ipython().run_line_magic('memit', 'sum_of_lists(1000000)')


# We see that this function uses about 100 MB of memory.
# 
# For a line-by-line description of memory use, we can use the ``%mprun`` magic.
# Unfortunately, this magic works only for functions defined in separate modules rather than the notebook itself, so we'll start by using the ``%%file`` magic to create a simple module called ``mprun_demo.py``, which contains our ``sum_of_lists`` function, with one addition that will make our memory profiling results more clear:

# In[14]:


get_ipython().run_cell_magic('file', 'mprun_demo.py', 'def sum_of_lists(N):\n    total = 0\n    for i in range(5):\n        L = [j ^ (j >> i) for j in range(N)]\n        total += sum(L)\n        del L # remove reference to L\n    return total')


# We can now import the new version of this function and run the memory line profiler:

# In[15]:


from mprun_demo import sum_of_lists
get_ipython().run_line_magic('mprun', '-f sum_of_lists sum_of_lists(1000000)')


# The result, printed to the pager, gives us a summary of the memory use of the function, and looks something like this:
# ```
# Filename: ./mprun_demo.py
# 
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      4     71.9 MiB      0.0 MiB           L = [j ^ (j >> i) for j in range(N)]
# 
# 
# Filename: ./mprun_demo.py
# 
# Line #    Mem usage    Increment   Line Contents
# ================================================
#      1     39.0 MiB      0.0 MiB   def sum_of_lists(N):
#      2     39.0 MiB      0.0 MiB       total = 0
#      3     46.5 MiB      7.5 MiB       for i in range(5):
#      4     71.9 MiB     25.4 MiB           L = [j ^ (j >> i) for j in range(N)]
#      5     71.9 MiB      0.0 MiB           total += sum(L)
#      6     46.5 MiB    -25.4 MiB           del L # remove reference to L
#      7     39.1 MiB     -7.4 MiB       return total
# ```
# Here the ``Increment`` column tells us how much each line affects the total memory budget: observe that when we create and delete the list ``L``, we are adding about 25 MB of memory usage.
# This is on top of the background memory usage from the Python interpreter itself.
# 
# For more information on ``%memit`` and ``%mprun``, as well as their available options, use the IPython help functionality (i.e., type ``%memit?`` at the IPython prompt).

# <!--NAVIGATION-->
# < [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) | [Contents](Index.ipynb) | [More IPython Resources](01.08-More-IPython-Resources.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.07-Timing-and-Profiling.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
