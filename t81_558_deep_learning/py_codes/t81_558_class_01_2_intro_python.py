#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 1: Python Preliminaries**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 1 Material
# 
# * Part 1.1: Course Overview [[Video]](https://www.youtube.com/watch?v=v8QsRio8zUM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_1_overview.ipynb)
# * **Part 1.2: Introduction to Python** [[Video]](https://www.youtube.com/watch?v=czq5d53vKvo&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_2_intro_python.ipynb)
# * Part 1.3: Python Lists, Dictionaries, Sets and JSON [[Video]](https://www.youtube.com/watch?v=kcGx2I5akSs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_3_python_collections.ipynb)
# * Part 1.4: File Handling [[Video]](https://www.youtube.com/watch?v=FSuSLCMgCZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_4_python_files.ipynb)
# * Part 1.5: Functions, Lambdas, and Map/Reduce [[Video]](https://www.youtube.com/watch?v=jQH1ZCSj6Ng&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_5_python_functional.ipynb)

# # Part 1.2: Introduction to Python
# 
# Like most tutorials, we will begin by printing Hello World.

# In[ ]:


print("Hello World") 


# You can also leave commends in your code to explain what you are doing.  Comments can begin anywhere in a line.

# In[2]:


# Single line comment (this has no effect on your program)
print("Hello World") # Say hello


# The triple quote allows multiple lines.

# In[3]:


print("""Print
Multiple
Lines
""")


# Python strings (which are textual) are always between quotes (double quotes) or apostrophes (single quote). There is no difference, if you wish to use the single quote:

# In[4]:


print('Hello World')


# Numbers do not need quotes:

# In[5]:


print(42)


# Variables can hold string or number values.

# In[6]:


a = 10
b = "ten"
print(a)
print(b)


# Variables are values that are held in memory that can change as your program runs.  A variable always has a name, such as **a** or **b**.  The value of a variable can change.

# In[7]:


a = 10
print(a)
a = a + 1
print(a)


# You can mix strings and variables for printing.  This is called a formatted string.  The variables must be inside of the curley braces.

# In[8]:


a = 10
print(f'The value of a is {a}')


# You can also use f-strings with math (called an expression)

# In[9]:


a = 10
print(f'The value of a plus 5 is {a+5}')


# Python has many ways to print numbers, these are all correct.  However, for this course we will use f-strings.

# In[10]:


a = 5

print(f'a is {a}') # Preferred method for this course.
print('a is {}'.format(a))
print('a is ' + str(a))
print('a is %d' % (a))


# You can use if-statements to perform logic.  Notice the indents?  This is how Python defines blocks of code that are executed together.  A block usually begins after a colon and includes any lines at the same level of indent.

# In[11]:


a = 5
if a>5:
    print('The variable a is greater than 5.')
else:
    print('The variable a is not greater than 5')


# The following if statement has multiple levels.  It can be easy to not properly indent these levels, so be careful.  Also choose tabs or spaces, not both.  The elif means "else if".

# In[12]:


a = 5
b = 6

if a==5:
    print('The variable a is 5')
    if b==6:
        print('The variable b is also 6')
elif a==6:
    print('The variable a is 6')
    


# Count to 9 in Python. Use a **for** loop and a **range**

# In[13]:


for x in range(1, 10):  # If you ever see xrange, you are in Python 2
    print(x)  # If you ever see print x (no parenthesis), you are in Python 2


# ## Printing Numbers and Strings

# In[14]:


acc = 0
for x in range(1, 10):
    acc += x
    print(f"Adding {x}, sum so far is {acc}")

print(f"Final sum: {acc}")

