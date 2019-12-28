#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # HDFS

# Run an HDFS command:

# In[ ]:


# get_ipython().system('hdfs')


# Run a file system command on the file systems (FsShell):

# In[ ]:


# get_ipython().system('hdfs dfs')


# List the user's home directory:

# In[ ]:


# get_ipython().system('hdfs dfs -ls')


# List the HDFS root directory:

# In[ ]:


# get_ipython().system('hdfs dfs -ls /')


# Copy a local file to the user's directory on HDFS:

# In[ ]:


# get_ipython().system('hdfs dfs -put file.txt file.txt')


# Display the contents of the specified HDFS file:

# In[ ]:


# get_ipython().system('hdfs dfs -cat file.txt')


# Print the last 10 lines of the file to the terminal:

# In[ ]:


# get_ipython().system('hdfs dfs -cat file.txt | tail -n 10')


# View a directory and all of its files:

# In[ ]:


# get_ipython().system('hdfs dfs -cat dir/* | less')


# Copy an HDFS file to local:

# In[ ]:


# get_ipython().system('hdfs dfs -get file.txt file.txt')


# Create a directory on HDFS:

# In[ ]:


# get_ipython().system('hdfs dfs -mkdir dir')


# Recursively delete the specified directory and all of its contents:

# In[ ]:


# get_ipython().system('hdfs dfs -rm -r dir')


# Specify HDFS file in Spark (paths are relative to the user's home HDFS directory):

# In[ ]:


data = sc.textFile ("hdfs://hdfs-host:port/path/file.txt")

