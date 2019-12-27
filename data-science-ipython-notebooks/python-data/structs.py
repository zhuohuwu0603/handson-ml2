#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Data Structures
# 
# * tuple
# * list
# * dict
# * set

# ## tuple

# A tuple is a one dimensional, fixed-length, immutable sequence.
# 
# Create a tuple:

# In[1]:


tup = (1, 2, 3)
tup


# Convert to a tuple:

# In[2]:


list_1 = [1, 2, 3]
type(tuple(list_1))


# Create a nested tuple:

# In[3]:


nested_tup = ([1, 2, 3], (4, 5))
nested_tup


# Access a tuple's elements by index O(1):

# In[4]:


nested_tup[0]


# Although tuples are immutable, their contents can contain mutable objects.  
# 
# Modify a tuple's contents:

# In[5]:


nested_tup[0].append(4)
nested_tup[0]


# Concatenate tuples by creating a new tuple and copying objects:

# In[6]:


(1, 3, 2) + (4, 5, 6)


# Multiply tuples to copy references to objects (objects themselves are not copied):

# In[7]:


('foo', 'bar') * 2


# Unpack tuples:

# In[8]:


a, b = nested_tup
a, b


# Unpack nested tuples:

# In[9]:


(a, b, c, d), (e, f) = nested_tup
a, b, c, d, e, f


# A common use of variable unpacking is when iterating over sequences of tuples or lists:

# In[10]:


seq = [( 1, 2, 3), (4, 5, 6), (7, 8, 9)] 
for a, b, c in seq: 
    print(a, b, c)


# ## list

# A list is a one dimensional, variable-length, mutable sequence.
# 
# Create a list:

# In[11]:


list_1 = [1, 2, 3]
list_1


# Convert to a list:

# In[12]:


type(list(tup))


# Create a nested list:

# In[13]:


nested_list = [(1, 2, 3), [4, 5]]
nested_list


# Access a list's elements by index O(1):

# In[14]:


nested_list[1]


# Append an element to a list O(1):

# In[15]:


nested_list.append(6)
nested_list


# Insert an element to a list at a specific index (note that insert is expensive as it has to shift subsequent elements O(n)):

# In[16]:


nested_list.insert(0, 'start')
nested_list


# Pop is expensive as it has to shift subsequent elements O(n).  The operation is O(1) if pop is used for the last element.
# 
# Remove and return an element from a specified index:

# In[17]:


nested_list.pop(0)
nested_list


# Locates the first such value and remove it O(n):

# In[18]:


nested_list.remove((1, 2, 3))
nested_list


# Check if a list contains a value O(n):

# In[19]:


6 in nested_list


# Concatenate lists by creating a new list and copying objects:

# In[20]:


[1, 3, 2] + [4, 5, 6]


# Extend a list by appending elements (faster than concatenating lists, as it does not have to create a new list):

# In[21]:


nested_list.extend([7, 8, 9])
nested_list


# ## dict

# A dict is also known as a hash map or associative array.  A dict is a mutable collection of key-value pairs.
# 
# Note: Big O complexities are listed as average case, with most worst case complexities being O(n).
# 
# Create a dict:

# In[22]:


dict_1 = { 'a' : 'foo', 'b' : [0, 1, 2, 3] }
dict_1


# Access a dict's elements by index O(1)

# In[23]:


dict_1['b']


# Insert or set a dict's elements by index O(1):

# In[24]:


dict_1[5] = 'bar'
dict_1


# Check if a dict contains a key O(1):

# In[25]:


5 in dict_1


# Delete a value from a dict O(1):

# In[26]:


dict_2 = dict(dict_1)
del dict_2[5]
dict_2


# Remove and return an element from a specified index O(1):

# In[27]:


value = dict_2.pop('b')
print(value)
print(dict_2)


# Get or pop can be called with a default value if the key is not found.  By default, get() will return None and pop() will throw an exception if the key is not found.

# In[28]:


value = dict_1.get('z', 0)
value


# Return a default value if the key is not found:

# In[29]:


print(dict_1.setdefault('b', None))
print(dict_1.setdefault('z', None))


# By contrast to setdefault(), defaultdict lets you specify the default when the container is initialized, which works well if the default is appropriate for all keys:

# In[30]:


from collections import defaultdict

seq = ['foo', 'bar', 'baz']
first_letter = defaultdict(list)
for elem in seq:
    first_letter[elem[0]].append(elem)
first_letter


# dict keys must be "hashable", i.e. they must be immutable objects like scalars (int, float, string) or tuples whose objects are all immutable.  Lists are mutable and therefore are not hashable, although you can convert the list portion to a tuple as a quick fix.

# In[31]:


print(hash('string'))
print(hash((1, 2, (3, 4))))


# Get the list of keys in no particular order (although keys() outputs the keys in the same order).  In Python 3, keys() returns an iterator instead of a list.

# In[32]:


dict_1.keys()


# Get the list of values in no particular order (although values() outputs the keys in the same order).  In Python 3, values() returns a [view object](https://docs.python.org/3/library/stdtypes.html?highlight=dictview#dictionary-view-objects) instead of a list.

# In[33]:


dict_1.values()


# Iterate through a dictionary's keys and values:

# In[34]:


for key, value in dict_1.items():
    print key, value


# Merge one dict into another:

# In[35]:


dict_1.update({'e' : 'elephant', 'f' : 'fish'})
dict_1


# Pair up two sequences element-wise in a dict:

# In[36]:


mapping = dict(zip(range(7), reversed(range(7))))
mapping


# ## set

# A set is an unordered sequence of unique elements.  
# 
# Create a set:

# In[37]:


set_1 = set([0, 1, 2, 3, 4, 5])
set_1


# In[38]:


set_2 = {1, 2, 3, 5, 8, 13}
set_2


# Sets support set operations like union, intersection, difference, and symmetric difference.

# Union O(len(set_1) + len(set_2)):

# In[39]:


set_1 | set_2


# Intersection O(min(len(set_1), len(set_2)):

# In[40]:


set_1 & set_2


# Difference O(len(set_1)):

# In[41]:


set_1 - set_2


# Symmetric Difference O(len(set_1)):

# In[42]:


set_1 ^ set_2


# Subset O(len(set_3)):

# In[43]:


set_3 = {1, 2, 3}
set_3.issubset(set_2)


# Superset O(len(set_3)):

# In[44]:


set_2.issuperset(set_3)


# Equal O(min(len(set_1), len(set_2)):

# In[45]:


{1, 2, 3} == {3, 2, 1}

