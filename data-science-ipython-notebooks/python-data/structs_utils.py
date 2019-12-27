#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Data Structure Utilities

# * slice
# * range and xrange
# * bisect
# * sort
# * sorted
# * reversed
# * enumerate
# * zip
# * list comprehensions

# ## slice

# Slice selects a section of list types (arrays, tuples, NumPy arrays) with its arguments [start:end]: start is included, end is not.  The number of elements in the result is stop - end.

# ![alt text](http://www.nltk.org/images/string-slicing.png)
# 
# Image source: http://www.nltk.org/images/string-slicing.png

# Slice 4 elements starting at index 6 and ending at index 9:

# In[1]:


seq = 'Monty Python'
seq[6:10]


# Omit start to default to start of the sequence:

# In[2]:


seq[:5]


# Omit end to default to end of the sequence:

# In[3]:


seq[6:]


# Negative indices slice relative to the end:

# In[4]:


seq[-12:-7]


# Slice can also take a step [start:end:step].
# 
# Get every other element:

# In[5]:


seq[::2]


# Passing -1 for the step reverses the list or tuple:

# In[6]:


seq[::-1]


# You can assign elements to a slice (note the slice range does not have to equal number of elements to assign):

# In[7]:


seq = [1, 1, 2, 3, 5, 8, 13]
seq[5:] = ['H', 'a', 'l', 'l']
seq


# Compare the output of assigning into a slice (above) versus the output of assigning into an index (below):

# In[8]:


seq = [1, 1, 2, 3, 5, 8, 13]
seq[5] = ['H', 'a', 'l', 'l']
seq


# ## range and xrange

# Generate a list of evenly spaced integers with range or xrange.  Note: range in Python 3 returns a generator and xrange is not available.
# 
# Generate 10 integers:

# In[9]:


range(10)


# Range can take start, stop, and step arguments:

# In[10]:


range(0, 20, 3)


# It is very common to iterate through sequences by index with range:

# In[11]:


seq = [1, 2, 3]
for i in range(len(seq)):
    val = seq[i]
    print(val)


# For longer ranges, xrange is recommended and is available in Python 3 as range.  It returns an iterator that generates integers one by one rather than all at once and storing them in a large list.

# In[12]:


sum = 0
for i in xrange(100000):
    if i % 2 == 0:
        sum += 1
print(sum)


# ## bisect

# The bisect module does not check whether the list is sorted, as this check would be expensive O(n).  Using bisect on an unsorted list will not result in an error but could lead to incorrect results.

# In[13]:


import bisect


# Find the location where an element should be inserted to keep the list sorted:

# In[14]:


seq = [1, 2, 2, 3, 5, 13]
bisect.bisect(seq, 8)


# Insert an element into a location to keep the list sorted:

# In[15]:


bisect.insort(seq, 8)
seq


# ## sort

# Sort in-place O(n log n)

# In[16]:


seq = [1, 5, 3, 9, 7, 6]
seq.sort()
seq


# Sort by the secondary key of str length:

# In[17]:


seq = ['the', 'quick', 'brown', 'fox', 'jumps', 'over']
seq.sort(key=len)
seq


# ## sorted

# Return a new sorted list from the elements of a sequence O(n log n):

# In[18]:


sorted([2, 5, 1, 8, 7, 9])


# In[19]:


sorted('foo bar baz')


# It's common to get a sorted list of unique elements by combining sorted and set:

# In[20]:


seq = [2, 5, 1, 8, 7, 9, 9, 2, 5, 1, (4, 2), (1, 2), (1, 2)]
sorted(set(seq))


# ## reversed

# Iterate over the sequence elements in reverse order:

# In[21]:


list(reversed(seq))


# ## enumerate

# Get the index of a collection and the value:

# In[22]:


strings = ['foo', 'bar', 'baz']
for i, string in enumerate(strings):
    print(i, string)


# ## zip

# Pair up the elements of sequences to create a list of tuples:

# In[23]:


seq_1 = [1, 2, 3]
seq_2 = ['foo', 'bar', 'baz']
zip(seq_1, seq_2)


# Zip takes an arbitrary number of sequences.  The number of elements it produces is determined by the shortest sequence:

# In[24]:


seq_3 = [True, False]
zip(seq_1, seq_2, seq_3)


# It is common to use zip for simultaneously iterating over multiple sequences combined with enumerate:

# In[25]:


for i, (a, b) in enumerate(zip(seq_1, seq_2)):
    print('%d: %s, %s' % (i, a, b))


# Zip can unzip a zipped sequence, which you can think of as converting a list of rows into a list of columns:

# In[26]:


numbers = [(1, 'one'), (2, 'two'), (3, 'three')]
a, b = zip(*numbers)
a


# In[27]:


b


# ## List Comprehensions

# List comprehensions concisely form a new list by filtering the elements of a sequence and transforming the elements passing the filter.  List comprehensions take the form:
# ```python
# [expr for val in collection if condition]
# ```
# Which is equivalent to the following for loop:
# ```python
# result = []
# for val in collection:
#     if condition:
#         result.append(expr)
# ```

# Convert to upper case all strings that start with a 'b':

# In[28]:


strings = ['foo', 'bar', 'baz', 'f', 'fo', 'b', 'ba']
[x.upper() for x in strings if x[0] == 'b']


# List comprehensions can be nested:

# In[29]:


list_of_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
[x for tup in list_of_tuples for x in tup]


# ## Dict Comprehension

# A dict comprehension is similar to a list comprehension but returns a dict.
# 
# Create a mapping of strings and their locations in the list for strings that start with a 'b':

# In[30]:


{index : val for index, val in enumerate(strings) if val[0] == 'b'}


# ## Set Comprehension

# A set comprehension is similar to a list comprehension but returns a set.
# 
# Get the unique lengths of strings that start with a 'b':

# In[31]:


{len(x) for x in strings if x[0] == 'b'}

