#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Files
# 
# * Read a File
# * Write a File
# * Read and Write UTF-8

# ## Read a File
# 
# Open a file in read-only mode.<br\>
# Iterate over the file lines.  rstrip removes the EOL markers.<br\>

# In[1]:


old_file_path = 'type_util.py'
with open(old_file_path, 'r') as old_file:
    for line in old_file:
        print(line.rstrip())


# ## Write to a file
# 
# Create a new file overwriting any previous file with the same name, write text, then close the file:

# In[2]:


new_file_path = 'hello_world.txt'
with open(new_file_path, 'w') as new_file:
    new_file.write('hello world!')


# ## Read and Write UTF-8

# In[3]:


import codecs
with codecs.open("hello_world_new.txt", "a", "utf-8") as new_file:
    with codecs.open("hello_world.txt", "r", "utf-8") as old_file:                   
        for line in old_file:
            new_file.write(line + '\n')

