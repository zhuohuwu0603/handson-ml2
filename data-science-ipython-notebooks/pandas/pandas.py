#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Pandas
# 
# Credits: The following are notes taken while working through [Python for Data Analysis](http://www.amazon.com/Python-Data-Analysis-Wrangling-IPython/dp/1449319793) by Wes McKinney
# 
# * Series
# * DataFrame
# * Reindexing
# * Dropping Entries
# * Indexing, Selecting, Filtering
# * Arithmetic and Data Alignment
# * Function Application and Mapping
# * Sorting and Ranking
# * Axis Indices with Duplicate Values
# * Summarizing and Computing Descriptive Statistics
# * Cleaning Data (Under Construction)
# * Input and Output (Under Construction)

# In[1]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# ## Series
# 
# A Series is a one-dimensional array-like object containing an array of data and an associated array of data labels.  The data can be any NumPy data type and the labels are the Series' index.

# Create a Series:

# In[2]:


ser_1 = Series([1, 1, 2, -3, -5, 8, 13])
ser_1


# Get the array representation of a Series:

# In[3]:


ser_1.values


# Index objects are immutable and hold the axis labels and metadata such as names and axis names.
# 
# Get the index of the Series:

# In[4]:


ser_1.index


# Create a Series with a custom index:

# In[5]:


ser_2 = Series([1, 1, 2, -3, -5], index=['a', 'b', 'c', 'd', 'e'])
ser_2


# Get a value from a Series:

# In[6]:


ser_2[4] == ser_2['e']


# Get a set of values from a Series by passing in a list:

# In[7]:


ser_2[['c', 'a', 'b']]


# Get values great than 0:

# In[8]:


ser_2[ser_2 > 0]


# Scalar multiply:

# In[9]:


ser_2 * 2


# Apply a numpy math function:

# In[10]:


import numpy as np
np.exp(ser_2)


# A Series is like a fixed-length, ordered dict.  
# 
# Create a series by passing in a dict:

# In[11]:


dict_1 = {'foo' : 100, 'bar' : 200, 'baz' : 300}
ser_3 = Series(dict_1)
ser_3


# Re-order a Series by passing in an index (indices not found are NaN):

# In[12]:


index = ['foo', 'bar', 'baz', 'qux']
ser_4 = Series(dict_1, index=index)
ser_4


# Check for NaN with the pandas method:

# In[13]:


pd.isnull(ser_4)


# Check for NaN with the Series method:

# In[14]:


ser_4.isnull()


# Series automatically aligns differently indexed data in arithmetic operations:

# In[15]:


ser_3 + ser_4


# Name a Series:

# In[16]:


ser_4.name = 'foobarbazqux'


# Name a Series index:

# In[17]:


ser_4.index.name = 'label'


# In[18]:


ser_4


# Rename a Series' index in place:

# In[19]:


ser_4.index = ['fo', 'br', 'bz', 'qx']
ser_4


# ## DataFrame
# 
# A DataFrame is a tabular data structure containing an ordered collection of columns.  Each column can have a different type.  DataFrames have both row and column indices and is analogous to a dict of Series.  Row and column operations are treated roughly symmetrically.  Columns returned when indexing a DataFrame are views of the underlying data, not a copy.  To obtain a copy, use the Series' copy method.
# 
# Create a DataFrame:

# In[20]:


data_1 = {'state' : ['VA', 'VA', 'VA', 'MD', 'MD'],
          'year' : [2012, 2013, 2014, 2014, 2015],
          'pop' : [5.0, 5.1, 5.2, 4.0, 4.1]}
df_1 = DataFrame(data_1)
df_1


# Create a DataFrame specifying a sequence of columns:

# In[21]:


df_2 = DataFrame(data_1, columns=['year', 'state', 'pop'])
df_2


# Like Series, columns that are not present in the data are NaN:

# In[22]:


df_3 = DataFrame(data_1, columns=['year', 'state', 'pop', 'unempl'])
df_3


# Retrieve a column by key, returning a Series:
# 

# In[23]:


df_3['state']


# Retrive a column by attribute, returning a Series:

# In[24]:


df_3.year


# Retrieve a row by position:

# In[25]:


df_3.ix[0]


# Update a column by assignment:

# In[26]:


df_3['unempl'] = np.arange(5)
df_3


# Assign a Series to a column (note if assigning a list or array, the length must match the DataFrame, unlike a Series):

# In[27]:


unempl = Series([6.0, 6.0, 6.1], index=[2, 3, 4])
df_3['unempl'] = unempl
df_3


# Assign a new column that doesn't exist to create a new column:

# In[28]:


df_3['state_dup'] = df_3['state']
df_3


# Delete a column:

# In[29]:


del df_3['state_dup']
df_3


# Create a DataFrame from a nested dict of dicts (the keys in the inner dicts are unioned and sorted to form the index in the result, unless an explicit index is specified):

# In[30]:


pop = {'VA' : {2013 : 5.1, 2014 : 5.2},
       'MD' : {2014 : 4.0, 2015 : 4.1}}
df_4 = DataFrame(pop)
df_4


# Transpose the DataFrame:

# In[31]:


df_4.T


# Create a DataFrame from a dict of Series:

# In[32]:


data_2 = {'VA' : df_4['VA'][1:],
          'MD' : df_4['MD'][2:]}
df_5 = DataFrame(data_2)
df_5


# Set the DataFrame index name:

# In[33]:


df_5.index.name = 'year'
df_5


# Set the DataFrame columns name:

# In[34]:


df_5.columns.name = 'state'
df_5


# Return the data contained in a DataFrame as a 2D ndarray:

# In[35]:


df_5.values


# If the columns are different dtypes, the 2D ndarray's dtype will accomodate all of the columns:

# In[36]:


df_3.values


# ## Reindexing

# Create a new object with the data conformed to a new index.  Any missing values are set to NaN.

# In[37]:


df_3


# Reindexing rows returns a new frame with the specified index:

# In[38]:


df_3.reindex(list(reversed(range(0, 6))))


# Missing values can be set to something other than NaN:

# In[39]:


df_3.reindex(range(6, 0), fill_value=0)


# Interpolate ordered data like a time series:

# In[40]:


ser_5 = Series(['foo', 'bar', 'baz'], index=[0, 2, 4])


# In[41]:


ser_5.reindex(range(5), method='ffill')


# In[42]:


ser_5.reindex(range(5), method='bfill')


# Reindex columns:

# In[43]:


df_3.reindex(columns=['state', 'pop', 'unempl', 'year'])


# Reindex rows and columns while filling rows:

# In[44]:


df_3.reindex(index=list(reversed(range(0, 6))),
             fill_value=0,
             columns=['state', 'pop', 'unempl', 'year'])


# Reindex using ix:

# In[45]:


df_6 = df_3.ix[range(0, 7), ['state', 'pop', 'unempl', 'year']]
df_6


# ## Dropping Entries

# Drop rows from a Series or DataFrame:

# In[46]:


df_7 = df_6.drop([0, 1])
df_7


# Drop columns from a DataFrame:

# In[47]:


df_7 = df_7.drop('unempl', axis=1)
df_7


# ## Indexing, Selecting, Filtering

# Series indexing is similar to NumPy array indexing with the added bonus of being able to use the Series' index values.

# In[48]:


ser_2


# Select a value from a Series:

# In[49]:


ser_2[0] == ser_2['a']


# Select a slice from a Series:

# In[50]:


ser_2[1:4]


# Select specific values from a Series:

# In[51]:


ser_2[['b', 'c', 'd']]


# Select from a Series based on a filter:

# In[52]:


ser_2[ser_2 > 0]


# Select a slice from a Series with labels (note the end point is inclusive):

# In[53]:


ser_2['a':'b']


# Assign to a Series slice (note the end point is inclusive):

# In[54]:


ser_2['a':'b'] = 0
ser_2


# Pandas supports indexing into a DataFrame.

# In[55]:


df_6


# Select specified columns from a DataFrame:

# In[56]:


df_6[['pop', 'unempl']]


# Select a slice from a DataFrame:

# In[57]:


df_6[:2]


# Select from a DataFrame based on a filter:

# In[58]:


df_6[df_6['pop'] > 5]


# Perform a scalar comparison on a DataFrame:

# In[59]:


df_6 > 5


# Perform a scalar comparison on a DataFrame, retain the values that pass the filter:

# In[60]:


df_6[df_6 > 5]


# Select a slice of rows from a DataFrame (note the end point is inclusive):

# In[61]:


df_6.ix[2:3]


# Select a slice of rows from a specific column of a DataFrame:

# In[62]:


df_6.ix[0:2, 'pop']


# Select rows based on an arithmetic operation on a specific row:

# In[63]:


df_6.ix[df_6.unempl > 5.0]


# ## Arithmetic and Data Alignment

# Adding Series objects results in the union of index pairs if the pairs are not the same, resulting in NaN for indices that do not overlap:

# In[64]:


np.random.seed(0)
ser_6 = Series(np.random.randn(5),
               index=['a', 'b', 'c', 'd', 'e'])
ser_6


# In[65]:


np.random.seed(1)
ser_7 = Series(np.random.randn(5),
               index=['a', 'c', 'e', 'f', 'g'])
ser_7


# In[66]:


ser_6 + ser_7


# Set a fill value instead of NaN for indices that do not overlap:

# In[67]:


ser_6.add(ser_7, fill_value=0)


# Adding DataFrame objects results in the union of index pairs for rows and columns if the pairs are not the same, resulting in NaN for indices that do not overlap:

# In[68]:


np.random.seed(0)
df_8 = DataFrame(np.random.rand(9).reshape((3, 3)),
                 columns=['a', 'b', 'c'])
df_8


# In[69]:


np.random.seed(1)
df_9 = DataFrame(np.random.rand(9).reshape((3, 3)),
                 columns=['b', 'c', 'd'])
df_9


# In[70]:


df_8 + df_9


# Set a fill value instead of NaN for indices that do not overlap:

# In[71]:


df_10 = df_8.add(df_9, fill_value=0)
df_10


# Like NumPy, pandas supports arithmetic operations between DataFrames and Series.
# 
# Match the index of the Series on the DataFrame's columns, broadcasting down the rows:

# In[72]:


ser_8 = df_10.ix[0]
df_11 = df_10 - ser_8
df_11


# Match the index of the Series on the DataFrame's columns, broadcasting down the rows and union the indices that do not match:

# In[73]:


ser_9 = Series(range(3), index=['a', 'd', 'e'])
ser_9


# In[74]:


df_11 - ser_9


# Broadcast over the columns and match the rows (axis=0) by using an arithmetic method:

# In[75]:


df_10


# In[76]:


ser_10 = Series([100, 200, 300])
ser_10


# In[77]:


df_10.sub(ser_10, axis=0)


# ## Function Application and Mapping

# NumPy ufuncs (element-wise array methods) operate on pandas objects:

# In[78]:


df_11 = np.abs(df_11)
df_11


# Apply a function on 1D arrays to each column:

# In[79]:


func_1 = lambda x: x.max() - x.min()
df_11.apply(func_1)


# Apply a function on 1D arrays to each row:

# In[80]:


df_11.apply(func_1, axis=1)


# Apply a function and return a DataFrame:

# In[81]:


func_2 = lambda x: Series([x.min(), x.max()], index=['min', 'max'])
df_11.apply(func_2)


# Apply an element-wise Python function to a DataFrame:

# In[82]:


func_3 = lambda x: '%.2f' %x
df_11.applymap(func_3)


# Apply an element-wise Python function to a Series:

# In[83]:


df_11['a'].map(func_3)


# ## Sorting and Ranking

# In[84]:


ser_4


# Sort a Series by its index:

# In[85]:


ser_4.sort_index()


# Sort a Series by its values:

# In[86]:


ser_4.sort_values()


# In[87]:


df_12 = DataFrame(np.arange(12).reshape((3, 4)),
                  index=['three', 'one', 'two'],
                  columns=['c', 'a', 'b', 'd'])
df_12


# Sort a DataFrame by its index:

# In[88]:


df_12.sort_index()


# Sort a DataFrame by columns in descending order:

# In[89]:


df_12.sort_index(axis=1, ascending=False)


# Sort a DataFrame's values by column:

# In[90]:


df_12.sort_values(by=['d', 'c'])


# Ranking is similar to numpy.argsort except that ties are broken by assigning each group the mean rank:

# In[91]:


ser_11 = Series([7, -5, 7, 4, 2, 0, 4, 7])
ser_11 = ser_11.sort_values()
ser_11


# In[92]:


ser_11.rank()


# Rank a Series according to when they appear in the data:

# In[93]:


ser_11.rank(method='first')


# Rank a Series in descending order, using the maximum rank for the group:

# In[94]:


ser_11.rank(ascending=False, method='max')


# DataFrames can rank over rows or columns.

# In[95]:


df_13 = DataFrame({'foo' : [7, -5, 7, 4, 2, 0, 4, 7],
                   'bar' : [-5, 4, 2, 0, 4, 7, 7, 8],
                   'baz' : [-1, 2, 3, 0, 5, 9, 9, 5]})
df_13


# Rank a DataFrame over rows:

# In[96]:


df_13.rank()


# Rank a DataFrame over columns:

# In[97]:


df_13.rank(axis=1)


# ## Axis Indexes with Duplicate Values

# Labels do not have to be unique in Pandas:

# In[98]:


ser_12 = Series(range(5), index=['foo', 'foo', 'bar', 'bar', 'baz'])
ser_12


# In[99]:


ser_12.index.is_unique


# Select Series elements:

# In[100]:


ser_12['foo']


# Select DataFrame elements:

# In[101]:


df_14 = DataFrame(np.random.randn(5, 4),
                  index=['foo', 'foo', 'bar', 'bar', 'baz'])
df_14


# In[102]:


df_14.ix['bar']


# ## Summarizing and Computing Descriptive Statistics

# Unlike NumPy arrays, Pandas descriptive statistics automatically exclude missing data.  NaN values are excluded unless the entire row or column is NA.

# In[103]:


df_6


# In[104]:


df_6.sum()


# Sum over the rows:

# In[105]:


df_6.sum(axis=1)


# Account for NaNs:

# In[106]:


df_6.sum(axis=1, skipna=False)


# ## Cleaning Data (Under Construction)
# * Replace
# * Drop
# * Concatenate

# In[107]:


from pandas import Series, DataFrame
import pandas as pd


# Setup a DataFrame:

# In[108]:


data_1 = {'state' : ['VA', 'VA', 'VA', 'MD', 'MD'],
          'year' : [2012, 2013, 2014, 2014, 2015],
          'population' : [5.0, 5.1, 5.2, 4.0, 4.1]}
df_1 = DataFrame(data_1)
df_1


# ### Replace

# Replace all occurrences of a string with another string, in place (no copy):

# In[109]:


df_1.replace('VA', 'VIRGINIA', inplace=True)
df_1


# In a specified column, replace all occurrences of a string with another string, in place (no copy):

# In[110]:


df_1.replace({'state' : { 'MD' : 'MARYLAND' }}, inplace=True)
df_1


# ### Drop

# Drop the 'population' column and return a copy of the DataFrame:

# In[111]:


df_2 = df_1.drop('population', axis=1)
df_2


# ### Concatenate

# Concatenate two DataFrames:

# In[112]:


data_2 = {'state' : ['NY', 'NY', 'NY', 'FL', 'FL'],
          'year' : [2012, 2013, 2014, 2014, 2015],
          'population' : [6.0, 6.1, 6.2, 3.0, 3.1]}
df_3 = DataFrame(data_2)
df_3


# In[113]:


df_4 = pd.concat([df_1, df_3])
df_4


# ## Input and Output (Under Construction)
# * Reading
# * Writing

# In[114]:


from pandas import Series, DataFrame
import pandas as pd


# ### Reading

# Read data from a CSV file into a DataFrame (use sep='\t' for TSV):

# In[115]:


df_1 = pd.read_csv("../data/ozone.csv")


# Get a summary of the DataFrame:

# In[116]:


df_1.describe()


# List the first five rows of the DataFrame:

# In[117]:


df_1.head()


# ### Writing

# Create a copy of the CSV file, encoded in UTF-8 and hiding the index and header labels:

# In[118]:


df_1.to_csv('../data/ozone_copy.csv', 
            encoding='utf-8', 
            index=False, 
            header=False)


# View the data directory:

# In[119]:


get_ipython().system('ls -l ../data/')

