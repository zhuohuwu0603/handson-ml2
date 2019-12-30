#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 2: Python for Machine Learning**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 2 Material
# 
# Main video lecture:
# 
# * Part 2.1: Introduction to Pandas [[Video]](https://www.youtube.com/watch?v=bN4UuCBdpZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_1_python_pandas.ipynb)
# * Part 2.2: Categorical Values [[Video]](https://www.youtube.com/watch?v=4a1odDpG0Ho&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_2_pandas_cat.ipynb)
# * **Part 2.3: Grouping, Sorting, and Shuffling in Python Pandas** [[Video]](https://www.youtube.com/watch?v=YS4wm5gD8DM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_3_pandas_grouping.ipynb)
# * Part 2.4: Using Apply and Map in Pandas for Keras [[Video]](https://www.youtube.com/watch?v=XNCEZ4WaPBY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_4_pandas_functional.ipynb)
# * Part 2.5: Feature Engineering in Pandas for Deep Learning in Keras [[Video]](https://www.youtube.com/watch?v=BWPTj4_Mi9E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_5_pandas_features.ipynb)

# # Part 2.3: Grouping, Sorting, and Shuffling  
# 
# ### Shuffling a Dataset
# The following code is used to shuffle and reindex a data set.  A random seed can be used to produce a consistent shuffling of the data set.

# In[1]:


import os
import pandas as pd
import numpy as np

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])

#np.random.seed(42) # Uncomment this line to get the same shuffle each time
df = df.reindex(np.random.permutation(df.index))
df.reset_index(inplace=True, drop=True)
print(df[0:10])


# ### Sorting a Data Set
# 
# Data sets can also be sorted.  This code sorts the MPG dataset by name and displays the first car.

# In[2]:


import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])

df = df.sort_values(by='name', ascending=True)
print(f"The first car is: {df['name'].iloc[0]}")
print(df[0:5])


# ### Grouping a Data Set
# 
# Grouping is a common operation on data sets.  In SQL, this operation is referred to as "GROUP BY".  Grouping is used to summarize data.  Because of this summarization the row could will either stay the same or more likely shrink after a grouping is applied.
# 
# The Auto MPG dataset is used to demonstrate grouping.

# In[3]:


import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])
print(df[0:5])


# The above data set can be used with group to perform summaries.  For example, the following code will group cylinders by the average (mean).  This code will provide the grouping.  In addition to mean, other aggregating functions, such as **sum** or **count** can be used. 

# In[4]:


g = df.groupby('cylinders')['mpg'].mean()
g


# It might be useful to have these **mean** values as a dictionary.

# In[5]:


d = g.to_dict()
d


# This allows you to quickly access an individual element, such as to lookup the mean for 6 cylinders.  This is used in target encoding, which is presented in this module.

# In[6]:


d[6]


# The code below shows how to count the number of rows that match each cylinder count.

# In[7]:


df.groupby('cylinders')['mpg'].count().to_dict()

