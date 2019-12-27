#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 
# *No changes were made to the contents of this notebook from the original.*

# <!--NAVIGATION-->
# < [Sorting Arrays](02.08-Sorting.ipynb) | [Contents](Index.ipynb) | [Data Manipulation with Pandas](03.00-Introduction-to-Pandas.ipynb) >

# # Structured Data: NumPy's Structured Arrays

# While often our data can be well represented by a homogeneous array of values, sometimes this is not the case. This section demonstrates the use of NumPy's *structured arrays* and *record arrays*, which provide efficient storage for compound, heterogeneous data.  While the patterns shown here are useful for simple operations, scenarios like this often lend themselves to the use of Pandas ``Dataframe``s, which we'll explore in [Chapter 3](03.00-Introduction-to-Pandas.ipynb).

# In[1]:


import numpy as np


# Imagine that we have several categories of data on a number of people (say, name, age, and weight), and we'd like to store these values for use in a Python program.
# It would be possible to store these in three separate arrays:

# In[2]:


name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]


# But this is a bit clumsy. There's nothing here that tells us that the three arrays are related; it would be more natural if we could use a single structure to store all of this data.
# NumPy can handle this through structured arrays, which are arrays with compound data types.
# 
# Recall that previously we created a simple array using an expression like this:

# In[3]:


x = np.zeros(4, dtype=int)


# We can similarly create a structured array using a compound data type specification:

# In[4]:


# Use a compound data type for structured arrays
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)


# Here ``'U10'`` translates to "Unicode string of maximum length 10," ``'i4'`` translates to "4-byte (i.e., 32 bit) integer," and ``'f8'`` translates to "8-byte (i.e., 64 bit) float."
# We'll discuss other options for these type codes in the following section.
# 
# Now that we've created an empty container array, we can fill the array with our lists of values:

# In[5]:


data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)


# As we had hoped, the data is now arranged together in one convenient block of memory.
# 
# The handy thing with structured arrays is that you can now refer to values either by index or by name:

# In[6]:


# Get all names
data['name']


# In[7]:


# Get first row of data
data[0]


# In[8]:


# Get the name from the last row
data[-1]['name']


# Using Boolean masking, this even allows you to do some more sophisticated operations such as filtering on age:

# In[9]:


# Get names where age is under 30
data[data['age'] < 30]['name']


# Note that if you'd like to do any operations that are any more complicated than these, you should probably consider the Pandas package, covered in the next chapter.
# As we'll see, Pandas provides a ``Dataframe`` object, which is a structure built on NumPy arrays that offers a variety of useful data manipulation functionality similar to what we've shown here, as well as much, much more.

# ## Creating Structured Arrays
# 
# Structured array data types can be specified in a number of ways.
# Earlier, we saw the dictionary method:

# In[10]:


np.dtype({'names':('name', 'age', 'weight'),
          'formats':('U10', 'i4', 'f8')})


# For clarity, numerical types can be specified using Python types or NumPy ``dtype``s instead:

# In[11]:


np.dtype({'names':('name', 'age', 'weight'),
          'formats':((np.str_, 10), int, np.float32)})


# A compound type can also be specified as a list of tuples:

# In[12]:


np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])


# If the names of the types do not matter to you, you can specify the types alone in a comma-separated string:

# In[13]:


np.dtype('S10,i4,f8')


# The shortened string format codes may seem confusing, but they are built on simple principles.
# The first (optional) character is ``<`` or ``>``, which means "little endian" or "big endian," respectively, and specifies the ordering convention for significant bits.
# The next character specifies the type of data: characters, bytes, ints, floating points, and so on (see the table below).
# The last character or characters represents the size of the object in bytes.
# 
# | Character        | Description           | Example                             |
# | ---------        | -----------           | -------                             | 
# | ``'b'``          | Byte                  | ``np.dtype('b')``                   |
# | ``'i'``          | Signed integer        | ``np.dtype('i4') == np.int32``      |
# | ``'u'``          | Unsigned integer      | ``np.dtype('u1') == np.uint8``      |
# | ``'f'``          | Floating point        | ``np.dtype('f8') == np.int64``      |
# | ``'c'``          | Complex floating point| ``np.dtype('c16') == np.complex128``|
# | ``'S'``, ``'a'`` | String                | ``np.dtype('S5')``                  |
# | ``'U'``          | Unicode string        | ``np.dtype('U') == np.str_``        |
# | ``'V'``          | Raw data (void)       | ``np.dtype('V') == np.void``        |

# ## More Advanced Compound Types
# 
# It is possible to define even more advanced compound types.
# For example, you can create a type where each element contains an array or matrix of values.
# Here, we'll create a data type with a ``mat`` component consisting of a $3\times 3$ floating-point matrix:

# In[14]:


tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])


# Now each element in the ``X`` array consists of an ``id`` and a $3\times 3$ matrix.
# Why would you use this rather than a simple multidimensional array, or perhaps a Python dictionary?
# The reason is that this NumPy ``dtype`` directly maps onto a C structure definition, so the buffer containing the array content can be accessed directly within an appropriately written C program.
# If you find yourself writing a Python interface to a legacy C or Fortran library that manipulates structured data, you'll probably find structured arrays quite useful!

# ## RecordArrays: Structured Arrays with a Twist
# 
# NumPy also provides the ``np.recarray`` class, which is almost identical to the structured arrays just described, but with one additional feature: fields can be accessed as attributes rather than as dictionary keys.
# Recall that we previously accessed the ages by writing:

# In[15]:


data['age']


# If we view our data as a record array instead, we can access this with slightly fewer keystrokes:

# In[16]:


data_rec = data.view(np.recarray)
data_rec.age


# The downside is that for record arrays, there is some extra overhead involved in accessing the fields, even when using the same syntax. We can see this here:

# In[17]:


get_ipython().run_line_magic('timeit', "data['age']")
get_ipython().run_line_magic('timeit', "data_rec['age']")
get_ipython().run_line_magic('timeit', 'data_rec.age')


# Whether the more convenient notation is worth the additional overhead will depend on your own application.

# ## On to Pandas
# 
# This section on structured and record arrays is purposely at the end of this chapter, because it leads so well into the next package we will cover: Pandas.
# Structured arrays like the ones discussed here are good to know about for certain situations, especially in case you're using NumPy arrays to map onto binary data formats in C, Fortran, or another language.
# For day-to-day use of structured data, the Pandas package is a much better choice, and we'll dive into a full discussion of it in the chapter that follows.

# <!--NAVIGATION-->
# < [Sorting Arrays](02.08-Sorting.ipynb) | [Contents](Index.ipynb) | [Data Manipulation with Pandas](03.00-Introduction-to-Pandas.ipynb) >
