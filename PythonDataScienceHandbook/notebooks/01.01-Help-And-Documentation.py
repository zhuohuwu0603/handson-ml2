#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb) | [Contents](Index.ipynb) | [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.01-Help-And-Documentation.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # Help and Documentation in IPython

# If you read no other section in this chapter, read this one: I find the tools discussed here to be the most transformative contributions of IPython to my daily workflow.
# 
# When a technologically-minded person is asked to help a friend, family member, or colleague with a computer problem, most of the time it's less a matter of knowing the answer as much as knowing how to quickly find an unknown answer.
# In data science it's the same: searchable web resources such as online documentation, mailing-list threads, and StackOverflow answers contain a wealth of information, even (especially?) if it is a topic you've found yourself searching before.
# Being an effective practitioner of data science is less about memorizing the tool or command you should use for every possible situation, and more about learning to effectively find the information you don't know, whether through a web search engine or another means.
# 
# One of the most useful functions of IPython/Jupyter is to shorten the gap between the user and the type of documentation and search that will help them do their work effectively.
# While web searches still play a role in answering complicated questions, an amazing amount of information can be found through IPython alone.
# Some examples of the questions IPython can help answer in a few keystrokes:
# 
# - How do I call this function? What arguments and options does it have?
# - What does the source code of this Python object look like?
# - What is in this package I imported? What attributes or methods does this object have?
# 
# Here we'll discuss IPython's tools to quickly access this information, namely the ``?`` character to explore documentation, the ``??`` characters to explore source code, and the Tab key for auto-completion.

# ## Accessing Documentation with ``?``
# 
# The Python language and its data science ecosystem is built with the user in mind, and one big part of that is access to documentation.
# Every Python object contains the reference to a string, known as a *doc string*, which in most cases will contain a concise summary of the object and how to use it.
# Python has a built-in ``help()`` function that can access this information and prints the results.
# For example, to see the documentation of the built-in ``len`` function, you can do the following:
# 
# ```ipython
# In [1]: help(len)
# Help on built-in function len in module builtins:
# 
# len(...)
#     len(object) -> integer
#     
#     Return the number of items of a sequence or mapping.
# ```
# 
# Depending on your interpreter, this information may be displayed as inline text, or in some separate pop-up window.

# Because finding help on an object is so common and useful, IPython introduces the ``?`` character as a shorthand for accessing this documentation and other relevant information:
# 
# ```ipython
# In [2]: len?
# Type:        builtin_function_or_method
# String form: <built-in function len>
# Namespace:   Python builtin
# Docstring:
# len(object) -> integer
# 
# Return the number of items of a sequence or mapping.
# ```

# This notation works for just about anything, including object methods:
# 
# ```ipython
# In [3]: L = [1, 2, 3]
# In [4]: L.insert?
# Type:        builtin_function_or_method
# String form: <built-in method insert of list object at 0x1024b8ea8>
# Docstring:   L.insert(index, object) -- insert object before index
# ```
# 
# or even objects themselves, with the documentation from their type:
# 
# ```ipython
# In [5]: L?
# Type:        list
# String form: [1, 2, 3]
# Length:      3
# Docstring:
# list() -> new empty list
# list(iterable) -> new list initialized from iterable's items
# ```

# Importantly, this will even work for functions or other objects you create yourself!
# Here we'll define a small function with a docstring:
# 
# ```ipython
# In [6]: def square(a):
#   ....:     """Return the square of a."""
#   ....:     return a ** 2
#   ....:
# ```
# 
# Note that to create a docstring for our function, we simply placed a string literal in the first line.
# Because doc strings are usually multiple lines, by convention we used Python's triple-quote notation for multi-line strings.

# Now we'll use the ``?`` mark to find this doc string:
# 
# ```ipython
# In [7]: square?
# Type:        function
# String form: <function square at 0x103713cb0>
# Definition:  square(a)
# Docstring:   Return the square of a.
# ```
# 
# This quick access to documentation via docstrings is one reason you should get in the habit of always adding such inline documentation to the code you write!

# ## Accessing Source Code with ``??``
# Because the Python language is so easily readable, another level of insight can usually be gained by reading the source code of the object you're curious about.
# IPython provides a shortcut to the source code with the double question mark (``??``):
# 
# ```ipython
# In [8]: square??
# Type:        function
# String form: <function square at 0x103713cb0>
# Definition:  square(a)
# Source:
# def square(a):
#     "Return the square of a"
#     return a ** 2
# ```
# 
# For simple functions like this, the double question-mark can give quick insight into the under-the-hood details.

# If you play with this much, you'll notice that sometimes the ``??`` suffix doesn't display any source code: this is generally because the object in question is not implemented in Python, but in C or some other compiled extension language.
# If this is the case, the ``??`` suffix gives the same output as the ``?`` suffix.
# You'll find this particularly with many of Python's built-in objects and types, for example ``len`` from above:
# 
# ```ipython
# In [9]: len??
# Type:        builtin_function_or_method
# String form: <built-in function len>
# Namespace:   Python builtin
# Docstring:
# len(object) -> integer
# 
# Return the number of items of a sequence or mapping.
# ```
# 
# Using ``?`` and/or ``??`` gives a powerful and quick interface for finding information about what any Python function or module does.

# ## Exploring Modules with Tab-Completion
# 
# IPython's other useful interface is the use of the tab key for auto-completion and exploration of the contents of objects, modules, and name-spaces.
# In the examples that follow, we'll use ``<TAB>`` to indicate when the Tab key should be pressed.

# ### Tab-completion of object contents
# 
# Every Python object has various attributes and methods associated with it.
# Like with the ``help`` function discussed before, Python has a built-in ``dir`` function that returns a list of these, but the tab-completion interface is much easier to use in practice.
# To see a list of all available attributes of an object, you can type the name of the object followed by a period ("``.``") character and the Tab key:
# 
# ```ipython
# In [10]: L.<TAB>
# L.append   L.copy     L.extend   L.insert   L.remove   L.sort     
# L.clear    L.count    L.index    L.pop      L.reverse  
# ```
# 
# To narrow-down the list, you can type the first character or several characters of the name, and the Tab key will find the matching attributes and methods:
# 
# ```ipython
# In [10]: L.c<TAB>
# L.clear  L.copy   L.count  
# 
# In [10]: L.co<TAB>
# L.copy   L.count 
# ```
# 
# If there is only a single option, pressing the Tab key will complete the line for you.
# For example, the following will instantly be replaced with ``L.count``:
# 
# ```ipython
# In [10]: L.cou<TAB>
# 
# ```
# 
# Though Python has no strictly-enforced distinction between public/external attributes and private/internal attributes, by convention a preceding underscore is used to denote such methods.
# For clarity, these private methods and special methods are omitted from the list by default, but it's possible to list them by explicitly typing the underscore:
# 
# ```ipython
# In [10]: L._<TAB>
# L.__add__           L.__gt__            L.__reduce__
# L.__class__         L.__hash__          L.__reduce_ex__
# ```
# 
# For brevity, we've only shown the first couple lines of the output.
# Most of these are Python's special double-underscore methods (often nicknamed "dunder" methods).

# ### Tab completion when importing
# 
# Tab completion is also useful when importing objects from packages.
# Here we'll use it to find all possible imports in the ``itertools`` package that start with ``co``:
# ```
# In [10]: from itertools import co<TAB>
# combinations                   compress
# combinations_with_replacement  count
# ```
# Similarly, you can use tab-completion to see which imports are available on your system (this will change depending on which third-party scripts and modules are visible to your Python session):
# ```
# In [10]: import <TAB>
# Display all 399 possibilities? (y or n)
# Crypto              dis                 py_compile
# Cython              distutils           pyclbr
# ...                 ...                 ...
# difflib             pwd                 zmq
# 
# In [10]: import h<TAB>
# hashlib             hmac                http         
# heapq               html                husl         
# ```
# (Note that for brevity, I did not print here all 399 importable packages and modules on my system.)

# ### Beyond tab completion: wildcard matching
# 
# Tab completion is useful if you know the first few characters of the object or attribute you're looking for, but is little help if you'd like to match characters at the middle or end of the word.
# For this use-case, IPython provides a means of wildcard matching for names using the ``*`` character.
# 
# For example, we can use this to list every object in the namespace that ends with ``Warning``:
# 
# ```ipython
# In [10]: *Warning?
# BytesWarning                  RuntimeWarning
# DeprecationWarning            SyntaxWarning
# FutureWarning                 UnicodeWarning
# ImportWarning                 UserWarning
# PendingDeprecationWarning     Warning
# ResourceWarning
# ```
# 
# Notice that the ``*`` character matches any string, including the empty string.
# 
# Similarly, suppose we are looking for a string method that contains the word ``find`` somewhere in its name.
# We can search for it this way:
# 
# ```ipython
# In [10]: str.*find*?
# str.find
# str.rfind
# ```
# 
# I find this type of flexible wildcard search can be very useful for finding a particular command when getting to know a new package or reacquainting myself with a familiar one.

# <!--NAVIGATION-->
# < [IPython: Beyond Normal Python](01.00-IPython-Beyond-Normal-Python.ipynb) | [Contents](Index.ipynb) | [Keyboard Shortcuts in the IPython Shell](01.02-Shell-Keyboard-Shortcuts.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.01-Help-And-Documentation.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
