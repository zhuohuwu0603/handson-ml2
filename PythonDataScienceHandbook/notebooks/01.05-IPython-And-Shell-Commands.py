#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# 
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*

# <!--NAVIGATION-->
# < [Input and Output History](01.04-Input-Output-History.ipynb) | [Contents](Index.ipynb) | [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.05-IPython-And-Shell-Commands.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 

# # IPython and Shell Commands

# When working interactively with the standard Python interpreter, one of the frustrations is the need to switch between multiple windows to access Python tools and system command-line tools.
# IPython bridges this gap, and gives you a syntax for executing shell commands directly from within the IPython terminal.
# The magic happens with the exclamation point: anything appearing after ``!`` on a line will be executed not by the Python kernel, but by the system command-line.
# 
# The following assumes you're on a Unix-like system, such as Linux or Mac OSX.
# Some of the examples that follow will fail on Windows, which uses a different type of shell by default (though with the 2016 announcement of native Bash shells on Windows, soon this may no longer be an issue!).
# If you're unfamiliar with shell commands, I'd suggest reviewing the [Shell Tutorial](http://swcarpentry.github.io/shell-novice/) put together by the always excellent Software Carpentry Foundation.

# ## Quick Introduction to the Shell
# 
# A full intro to using the shell/terminal/command-line is well beyond the scope of this chapter, but for the uninitiated we will offer a quick introduction here.
# The shell is a way to interact textually with your computer.
# Ever since the mid 1980s, when Microsoft and Apple introduced the first versions of their now ubiquitous graphical operating systems, most computer users have interacted with their operating system through familiar clicking of menus and drag-and-drop movements.
# But operating systems existed long before these graphical user interfaces, and were primarily controlled through sequences of text input: at the prompt, the user would type a command, and the computer would do what the user told it to.
# Those early prompt systems are the precursors of the shells and terminals that most active data scientists still use today.
# 
# Someone unfamiliar with the shell might ask why you would bother with this, when many results can be accomplished by simply clicking on icons and menus.
# A shell user might reply with another question: why hunt icons and click menus when you can accomplish things much more easily by typing?
# While it might sound like a typical tech preference impasse, when moving beyond basic tasks it quickly becomes clear that the shell offers much more control of advanced tasks, though admittedly the learning curve can intimidate the average computer user.
# 
# As an example, here is a sample of a Linux/OSX shell session where a user explores, creates, and modifies directories and files on their system (``osx:~ $`` is the prompt, and everything after the ``$`` sign is the typed command; text that is preceded by a ``#`` is meant just as description, rather than something you would actually type in):
# 
# ```bash
# osx:~ $ echo "hello world"             # echo is like Python's print function
# hello world
# 
# osx:~ $ pwd                            # pwd = print working directory
# /home/jake                             # this is the "path" that we're sitting in
# 
# osx:~ $ ls                             # ls = list working directory contents
# notebooks  projects 
# 
# osx:~ $ cd projects/                   # cd = change directory
# 
# osx:projects $ pwd
# /home/jake/projects
# 
# osx:projects $ ls
# datasci_book   mpld3   myproject.txt
# 
# osx:projects $ mkdir myproject          # mkdir = make new directory
# 
# osx:projects $ cd myproject/
# 
# osx:myproject $ mv ../myproject.txt ./  # mv = move file. Here we're moving the
#                                         # file myproject.txt from one directory
#                                         # up (../) to the current directory (./)
# osx:myproject $ ls
# myproject.txt
# ```
# 
# Notice that all of this is just a compact way to do familiar operations (navigating a directory structure, creating a directory, moving a file, etc.) by typing commands rather than clicking icons and menus.
# Note that with just a few commands (``pwd``, ``ls``, ``cd``, ``mkdir``, and ``cp``) you can do many of the most common file operations.
# It's when you go beyond these basics that the shell approach becomes really powerful.

# ## Shell Commands in IPython
# 
# Any command that works at the command-line can be used in IPython by prefixing it with the ``!`` character.
# For example, the ``ls``, ``pwd``, and ``echo`` commands can be run as follows:
# 
# ```ipython
# In [1]: !ls
# myproject.txt
# 
# In [2]: !pwd
# /home/jake/projects/myproject
# 
# In [3]: !echo "printing from the shell"
# printing from the shell
# ```

# ## Passing Values to and from the Shell
# 
# Shell commands can not only be called from IPython, but can also be made to interact with the IPython namespace.
# For example, you can save the output of any shell command to a Python list using the assignment operator:
# 
# ```ipython
# In [4]: contents = !ls
# 
# In [5]: print(contents)
# ['myproject.txt']
# 
# In [6]: directory = !pwd
# 
# In [7]: print(directory)
# ['/Users/jakevdp/notebooks/tmp/myproject']
# ```
# 
# Note that these results are not returned as lists, but as a special shell return type defined in IPython:
# 
# ```ipython
# In [8]: type(directory)
# IPython.utils.text.SList
# ```
# 
# This looks and acts a lot like a Python list, but has additional functionality, such as
# the ``grep`` and ``fields`` methods and the ``s``, ``n``, and ``p`` properties that allow you to search, filter, and display the results in convenient ways.
# For more information on these, you can use IPython's built-in help features.

# Communication in the other direction–passing Python variables into the shell–is possible using the ``{varname}`` syntax:
# 
# ```ipython
# In [9]: message = "hello from Python"
# 
# In [10]: !echo {message}
# hello from Python
# ```
# 
# The curly braces contain the variable name, which is replaced by the variable's contents in the shell command.

# # Shell-Related Magic Commands
# 
# If you play with IPython's shell commands for a while, you might notice that you cannot use ``!cd`` to navigate the filesystem:
# 
# ```ipython
# In [11]: !pwd
# /home/jake/projects/myproject
# 
# In [12]: !cd ..
# 
# In [13]: !pwd
# /home/jake/projects/myproject
# ```
# 
# The reason is that shell commands in the notebook are executed in a temporary subshell.
# If you'd like to change the working directory in a more enduring way, you can use the ``%cd`` magic command:
# 
# ```ipython
# In [14]: %cd ..
# /home/jake/projects
# ```
# 
# In fact, by default you can even use this without the ``%`` sign:
# 
# ```ipython
# In [15]: cd myproject
# /home/jake/projects/myproject
# ```
# 
# This is known as an ``automagic`` function, and this behavior can be toggled with the ``%automagic`` magic function.
# 
# Besides ``%cd``, other available shell-like magic functions are ``%cat``, ``%cp``, ``%env``, ``%ls``, ``%man``, ``%mkdir``, ``%more``, ``%mv``, ``%pwd``, ``%rm``, and ``%rmdir``, any of which can be used without the ``%`` sign if ``automagic`` is on.
# This makes it so that you can almost treat the IPython prompt as if it's a normal shell:
# 
# ```ipython
# In [16]: mkdir tmp
# 
# In [17]: ls
# myproject.txt  tmp/
# 
# In [18]: cp myproject.txt tmp/
# 
# In [19]: ls tmp
# myproject.txt
# 
# In [20]: rm -r tmp
# ```
# 
# This access to the shell from within the same terminal window as your Python session means that there is a lot less switching back and forth between interpreter and shell as you write your Python code.

# <!--NAVIGATION-->
# < [Input and Output History](01.04-Input-Output-History.ipynb) | [Contents](Index.ipynb) | [Errors and Debugging](01.06-Errors-and-Debugging.ipynb) >
# 
# <a href="https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/01.05-IPython-And-Shell-Commands.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>
# 
