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
# * Part 2.3: Grouping, Sorting, and Shuffling in Python Pandas [[Video]](https://www.youtube.com/watch?v=YS4wm5gD8DM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_3_pandas_grouping.ipynb)
# * **Part 2.4: Using Apply and Map in Pandas for Keras** [[Video]](https://www.youtube.com/watch?v=XNCEZ4WaPBY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_4_pandas_functional.ipynb)
# * Part 2.5: Feature Engineering in Pandas for Deep Learning in Keras [[Video]](https://www.youtube.com/watch?v=BWPTj4_Mi9E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_5_pandas_features.ipynb)

# # Part 2.4: Apply and Map

# The **apply** and **map** functions can also be applied to Pandas **dataframes**.

# ### Using Map with Dataframes

# In[1]:


import os
import pandas as pd
import numpy as np

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])

print(df[0:10])


# In[2]:


df['origin_name'] = df['origin'].map({1: 'North America', 2: 'Europe', 3: 'Asia'})
print(df[0:50])


# ### Using Apply with Dataframes
# 
# If the **apply** function is directly executed on the data frame, the lambda function is called once per column or row, depending on the value of axis.  For axis = 1, rows are used. 
# 
# The following code calculates a series called **efficiency** that is the **displacement** divided by **horsepower**. 

# In[3]:


effi = df.apply(lambda x: x['displacement']/x['horsepower'], axis=1)
print(effi[0:10])


# ### Feature Engineering with Apply and Map

# In this section we will see how to calculate a complex feature using map, apply, and grouping.  The data set is the following CSV:
# 
# * https://www.irs.gov/pub/irs-soi/16zpallagi.csv 
# 
# This is US Government public data for "SOI Tax Stats - Individual Income Tax Statistics".  The primary website is here:
# 
# * https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2016-zip-code-data-soi 
# 
# Documentation describing this data is at the above link.
# 
# For this feature, we will attempt to estimate the adjusted gross income (AGI) for each of the zipcodes.  The data file contains many columns; however, you will only use the following:
# 
# * STATE - The state (e.g. MO)
# * zipcode - The zipcode (e.g. 63017)
# * agi_stub - Six different brackets of annual income (1 through 6) 
# * N1 - The number of tax returns for each of the agi_stubs
# 
# Note, the file will have 6 rows for each zipcode, for each of the agi_stub brackets. You can skip zipcodes with 0 or 99999.
# 
# We will create an output CSV with these columns; however, only one row per zip code. Calculate a weighted average of the income brackets. For example, the following 6 rows are present for 63017:
# 
# 
# |zipcode |agi_stub | N1 |
# |--|--|-- |
# |63017	 |1 | 4710 |
# |63017	 |2 | 2780 |
# |63017	 |3 | 2130 |
# |63017	 |4 | 2010 |
# |63017	 |5 | 5240 |
# |63017	 |6 | 3510 |
# 
# 
# We must combine these six rows into one.  For privacy reasons, AGI's are broken out into 6 buckets.  We need to combine the buckets and estimate the actual AGI of a zipcode. To do this, consider the values for N1:
# 
# * 1 = \$1 to \$25,000
# * 2 = \$25,000 to \$50,000
# * 3 = \$50,000 to \$75,000
# * 4 = \$75,000 to \$100,000
# * 5 = \$100,000 to \$200,000
# * 6 = \$200,000 or more
# 
# The median of each of these ranges is approximately:
# 
# * 1 = \$12,500
# * 2 = \$37,500
# * 3 = \$62,500 
# * 4 = \$87,500
# * 5 = \$112,500
# * 6 = \$212,500
# 
# Using this you can estimate 63017's average AGI as:
# 
# ```
# >>> totalCount = 4710 + 2780 + 2130 + 2010 + 5240 + 3510
# >>> totalAGI = 4710 * 12500 + 2780 * 37500 + 2130 * 62500 + 2010 * 87500 + 5240 * 112500 + 3510 * 212500
# >>> print(totalAGI / totalCount)
# 
# 88689.89205103042
# ```

# In[4]:


import pandas as pd

df=pd.read_csv('https://www.irs.gov/pub/irs-soi/16zpallagi.csv')


# First, we trim all zipcodes that are either 0 or 99999.  We also select the three fields that we need.

# In[5]:


df=df.loc[(df['zipcode']!=0) & (df['zipcode']!=99999),['STATE','zipcode','agi_stub','N1']]


# In[6]:


df


# We replace all of the **agi_stub** values with the correct median values with the **map** function.

# In[7]:


medians = {1:12500,2:37500,3:62500,4:87500,5:112500,6:212500}
df['agi_stub']=df.agi_stub.map(medians)


# In[8]:


df


# Next the dataframe is grouped by zip code.

# In[9]:


groups = df.groupby(by='zipcode')


# A lambda is applied across the groups and the AGI estimate is calculated.

# In[10]:


df = pd.DataFrame(groups.apply(lambda x:sum(x['N1']*x['agi_stub'])/sum(x['N1']))).reset_index()


# In[11]:


df


# The new agi_estimate column is renamed.

# In[12]:


df.columns = ['zipcode','agi_estimate']


# In[13]:


print(df[0:10])


# We can also see that our zipcode of 63017 gets the correct value.

# In[14]:


df[ df['zipcode']==63017 ]


# In[ ]:




