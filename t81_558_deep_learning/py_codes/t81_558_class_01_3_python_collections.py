#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 1: Python Preliminaries**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 1 Material
# 
# * Part 1.1: Course Overview [[Video]](https://www.youtube.com/watch?v=v8QsRio8zUM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_1_overview.ipynb)
# * Part 1.2: Introduction to Python [[Video]](https://www.youtube.com/watch?v=czq5d53vKvo&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_2_intro_python.ipynb)
# * **Part 1.3: Python Lists, Dictionaries, Sets and JSON** [[Video]](https://www.youtube.com/watch?v=kcGx2I5akSs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_3_python_collections.ipynb)
# * Part 1.4: File Handling [[Video]](https://www.youtube.com/watch?v=FSuSLCMgCZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_4_python_files.ipynb)
# * Part 1.5: Functions, Lambdas, and Map/Reduce [[Video]](https://www.youtube.com/watch?v=jQH1ZCSj6Ng&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_5_python_functional.ipynb)

# # Part 1.3: Python Lists, Dictionaries, Sets and JSON
# 
# Like most modern programming languages Python includes Lists and Dictionaries.  The syntax appearance of both of these is similar to JSON.  It is possible to include syntactically correct JSON inside of Python definitions.

# In[1]:


c = ['a', 'b', 'c', 'd']
print(c)


# Like many languages, Python has a for-each statement.  This allows you to loop over every element in a collection.

# In[2]:


# Iterate over a collection.
for s in c:
    print(s)


# The **enumerate** function is useful for enumerating over a collection and having access to the index of the element that we are currently on.

# In[3]:


# Iterate over a collection, and know where your index.  (Python is zero-based!)
for i,c in enumerate(c):
    print(f"{i}:{c}")


# A **list/array** can have multiple objects added to it, such as strings.  

# In[4]:


# Manually add items, lists allow duplicates
c = []
c.append('a')
c.append('b')
c.append('c')
c.append('c')
print(c)


# A **set** can include objects/strings, but there 

# In[5]:


# Manually add items, sets do not allow duplicates
# Sets add, lists append.  I find this annoying.
c = set()
c.add('a')
c.add('b')
c.add('c')
c.add('c')
print(c)


# A **list** can have items inserted or removed.  For insert, a index must be specified.

# In[6]:


# Insert
c = ['a', 'b', 'c']
c.insert(0, 'a0')
print(c)
# Remove
c.remove('b')
print(c)
# Remove at index
del c[0]
print(c)


# ## Maps/Dictionaries/Hash Tables
# 
# Many programming languages include the concept of a map, dictionary, or hash table.  These are all very related concepts.  This is essentially a collection of name value pairs.  

# In[7]:


d = {'name': "Jeff", 'address':"123 Main"}
print(d)
print(d['name'])

if 'name' in d:
    print("Name is defined")

if 'age' in d:
    print("age defined")
else:
    print("age undefined")


# You can also access the individual keys and values of a dictionary.

# In[8]:


d = {'name': "Jeff", 'address':"123 Main"}
# All of the keys
print(f"Key: {d.keys()}")

# All of the values
print(f"Values: {d.values()}")


# Dictionaries and lists can be combined. This syntax is closely related to [JSON](https://en.wikipedia.org/wiki/JSON).  Very complex data structures can be built this way.  While Python allows quotes (") and apostrophe (') for strings, JSON only allows double-quotes ("). 

# In[9]:


# Python list & map structures
customers = [
    {"name": "Jeff & Tracy Heaton", "pets": ["Wynton", "Cricket", "Hickory"]},
    {"name": "John Smith", "pets": ["rover"]},
    {"name": "Jane Doe"}
]

print(customers)

for customer in customers:
    print(f"{customer['name']}:{customer.get('pets', 'no pets')}")


# ## More Advanced Lists
# 
# Two lists can be zipped together.

# In[10]:


a = [1,2,3,4,5]
b = [5,4,3,2,1]

print(zip(a,b))


# To actually see it, convert it to a list.

# In[11]:


a = [1,2,3,4,5]
b = [5,4,3,2,1]

print(list(zip(a,b)))


# This can be used in a loop.

# In[12]:


a = [1,2,3,4,5]
b = [5,4,3,2,1]

for x,y in zip(a,b):
    print(f'{x} - {y}')


# Use a list enumerate function to track what index location a list element is at.

# In[13]:


a = ['one','two','three','four','five']
list(enumerate(a))


# This can be handy in a list when you need to know what index you are on.

# In[14]:


a = ['one','two','three','four','five']
for idx, item in enumerate(a):
    print(f'Index {idx} holds "{item}"')


# A comprehension can be used to dynamically build up a list.  The comprehension below counts from 0 to 9 and adds each value (multiplied by 10) to a list.

# In[15]:


lst = [x*10 for x in range(10)]
print(lst)


# A dictionary can also be a comprehension.  The general format for this is:  dict_variable = {key:value for (key,value) in dictonary.items()}
# 
# A common use for this is to build up an index to symbolic column names.

# In[16]:


text = ['col-zero','col-one', 'col-two', 'col-three']
lookup = {key:value for (value,key) in enumerate(text)}
print(lookup)


# This can be used to easily find the index of a column by name.

# In[17]:


print(f'The index of "col-two" is {lookup["col-two"]}')

