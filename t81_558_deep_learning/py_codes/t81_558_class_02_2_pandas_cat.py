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
# * **Part 2.2: Categorical Values** [[Video]](https://www.youtube.com/watch?v=4a1odDpG0Ho&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_2_pandas_cat.ipynb)
# * Part 2.3: Grouping, Sorting, and Shuffling in Python Pandas [[Video]](https://www.youtube.com/watch?v=YS4wm5gD8DM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_3_pandas_grouping.ipynb)
# * Part 2.4: Using Apply and Map in Pandas for Keras [[Video]](https://www.youtube.com/watch?v=XNCEZ4WaPBY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_4_pandas_functional.ipynb)
# * Part 2.5: Feature Engineering in Pandas for Deep Learning in Keras [[Video]](https://www.youtube.com/watch?v=BWPTj4_Mi9E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_5_pandas_features.ipynb)

# # Part 2.2: Categorical and Continuous Values
# 
# Neural networks require their input to be a fixed number of columns.  This is very similar to spreadsheet data.  This input must be completely numeric.  
# 
# It is important to represent the data in a way that the neural network can train from it.  In class 6, we will see even more ways to preprocess data.  For now, we will look at several of the most basic ways to transform data for a neural network.
# 
# Before we look at specific ways to preprocess data, it is important to consider four basic types of data, as defined by [Stanley Smith Stevens](https://en.wikipedia.org/wiki/Stanley_Smith_Stevens).  These are commonly referred to as the [levels of measure](https://en.wikipedia.org/wiki/Level_of_measurement):
# 
# * Character Data (strings)
#     * **Nominal** - Individual discrete items, no order. For example: color, zip code, shape.
#     * **Ordinal** - Individual discrete items that can be ordered.  For example: grade level, job title, Starbucks(tm) coffee size (tall, vente, grande) 
# * Numeric Data
#     * **Interval** - Numeric values, no defined start.  For example, temperature.  You would never say "yesterday was twice as hot as today".
#     * **Ratio** - Numeric values, clearly defined start.  For example, speed.  You would say that "The first car is going twice as fast as the second."

# ### Encoding Continuous Values
# 
# One common transformation is to normalize the inputs.  It is sometimes valuable to normalization numeric inputs to be put in a standard form so that two values can easily be compared.  Consider if a friend told you that he received a $10 discount.  Is this a good deal?  Maybe.  But the value is not normalized.  If your friend purchased a car, then the discount is not that good.  If your friend purchased dinner, this is a very good discount!
# 
# Percentages are a very common form of normalization.  If your friend tells you they got 10% off, we know that this is a better discount than 5%.  It does not matter how much the purchase price was.  One very common machine learning normalization is the Z-Score:
# 
# $z = \frac{x - \mu}{\sigma} $
# 
# To calculate the Z-Score you need to also calculate the mean($\mu$) and the standard deviation ($\sigma$).  The mean is calculated as follows:
# 
# $\mu = \bar{x} = \frac{x_1+x_2+\cdots +x_n}{n}$
# 
# The standard deviation is calculated as follows:
# 
# $\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2}, {\rm \ \ where\ \ } \mu = \frac{1}{N} \sum_{i=1}^N x_i$
# 
# The following Python code replaces the mpg with a z-score.  Cars with average MPG will be near zero, above zero is above average, and below zero is below average.  Z-Scores above/below -3/3 are very rare, these are outliers.

# In[1]:


import os
import pandas as pd
from scipy.stats import zscore

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

df['mpg'] = zscore(df['mpg'])
print(df[0:5])


# ### Encoding Categorical Values as Dummies
# The classic means of encoding categorical values is to make them dummy variables.  This is also called one-hot-encoding.  Consider the following data set.

# In[2]:


import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

print(df[0:5])


# In[3]:


areas = list(df['area'].unique())
print(f'Number of areas: {len(areas)}')
print(f'Areas: {areas}')


# There are four unique values in the areas column.  To encode these to dummy variables we would use four columns, each of which would represent one of the areas.  For each row, one column would have a value of one, the rest zeros.  This is why this type of encoding is sometimes called one-hot encoding.  The following code shows how you might encode the values "a" through "d".  The value A becomes [1,0,0,0] and the value B becomes [0,1,0,0].

# In[4]:


dummies = pd.get_dummies(['a','b','c','d'],prefix='area')
print(dummies)


# To encode the "area" column, we use the following.

# It is necessary to merge these dummies back into the data frame.  

# In[5]:


dummies = pd.get_dummies(df['area'],prefix='area')
print(dummies[0:10]) # Just show the first 10


# In[6]:


df = pd.concat([df,dummies],axis=1)


# Displaying select columns from the dataset we can see the dummy variables added.

# In[7]:


print(df[0:10][['id','job','area','income','area_a',
                  'area_b','area_c','area_d']])


# Usually, you will remove the original column ('area'), because it is the goal to get the dataframe to be entirely numeric for the neural network.

# In[8]:


df.drop('area', axis=1, inplace=True)
print(df[0:10][['id','job','income','area_a',
                  'area_b','area_c','area_d']])


# ### Target Encoding for Categoricals
# 
# Target encoding can sometimes increase the predictive power of a machine learning model.  However, it also greatly increases the risk of overfitting.  Because of this risk, care must be take if you are going to use this method.  It is a popular technique for Kaggle competitions.  
# 
# Generally, target encoding can only be used on a categorical feature when the output of the machine learning model is numeric (regression).
# 
# The concept of target encoding is actually very simple.  For each value 

# In[9]:


# Create a small sample dataset
import pandas as pd
import numpy as np

np.random.seed(43)
df = pd.DataFrame({
    'cont_9': np.random.rand(10)*100,
    'cat_0': ['dog'] * 5 + ['cat'] * 5,
    'cat_1': ['wolf'] * 9 + ['tiger'] * 1,
    'y': [1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
})

print(df)


# Rather than creating dummy variables for dog and cat, we would like to change it to a number. We could just use 0 for cat, 1 for dog.  However, we can encode more information than just that.  The simple 0 or 1 would also only work for one animal.  Consider what the mean target value is for cat and dog.

# In[10]:


means0 = df.groupby('cat_0')['y'].mean().to_dict()
means0


# The danger is that we are now using the target value for training.  This will potentially overfit.  The possibility of overfitting is even greater if there are a small number of a particular category.  To prevent this from happening, we use a weighting factor.  The stronger the weight the more than categories with a small number of values will tend towards the overall average of y, which is calculated as follows.

# In[11]:


df['y'].mean()


# The complete function for target encoding is given here.

# In[12]:


# Source: https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
def calc_smooth_mean(df1, df2, cat_name, target, weight):
    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())


# The following code encodes these two categories.

# In[13]:


WEIGHT = 5
df['cat_0_enc'] = calc_smooth_mean(df1=df, df2=None, cat_name='cat_0', target='y', weight=WEIGHT)
df['cat_1_enc'] = calc_smooth_mean(df1=df, df2=None, cat_name='cat_1', target='y', weight=WEIGHT)


# In[14]:


print(df)


# ### Encoding Categorical Values as Ordinal
# 
# Typically categoricals will be encoded as dummy variables.  However, there might be other techniques to convert categoricals to numeric. Any time there is an order to the categoricals, a number should be used.  Consider if you had a categorical that described the current education level of an individual.   
# 
# * Kindergarten (0)
# * First Grade (1)
# * Second Grade (2)
# * Third Grade (3)
# * Fourth Grade (4)
# * Fifth Grade (5)
# * Sixth Grade (6)
# * Seventh Grade (7)
# * Eighth Grade (8)
# * High School Freshman (9)
# * High School Sophomore (10)
# * High School Junior (11)
# * High School Senior (12)
# * College Freshman (13)
# * College Sophomore (14)
# * College Junior (15)
# * College Senior (16)
# * Graduate Student (17)
# * PhD Candidate (18)
# * Doctorate (19)
# * Post Doctorate (20)
# 
# The above list has 21 levels.  This would take 21 dummy variables. However, simply encoding this to dummies would lose the order information.  Perhaps the easiest approach would be to assign simply number them and assign the category a single number that is equal to the value in parenthesis above.  However, we might be able to do even better.  Graduate student is likely more than a year, so you might increase more than just one value.  

# In[ ]:




