#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 4: Training for Tabular Data**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 4 Material
# 
# * **Part 4.1: Encoding a Feature Vector for Keras Deep Learning** [[Video]](https://www.youtube.com/watch?v=Vxz-gfs9nMQ&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_04_1_feature_encode.ipynb)
# * Part 4.2: Keras Multiclass Classification for Deep Neural Networks with ROC and AUC [[Video]](https://www.youtube.com/watch?v=-f3bg9dLMks&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_04_2_multi_class.ipynb)
# * Part 4.3: Keras Regression for Deep Neural Networks with RMSE [[Video]](https://www.youtube.com/watch?v=wNhBUC6X5-E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_04_3_regression.ipynb)
# * Part 4.4: Backpropagation, Nesterov Momentum, and ADAM Neural Network Training [[Video]](https://www.youtube.com/watch?v=VbDg8aBgpck&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_04_4_backprop.ipynb)
# * Part 4.5: Neural Network RMSE and Log Loss Error Calculation from Scratch [[Video]](https://www.youtube.com/watch?v=wmQX1t2PHJc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_04_5_rmse_logloss.ipynb)

# # Part 4.1: Encoding a Feature Vector for Keras Deep Learning
# 
# Neural networks can accept many types of data.  We will begin with tabular data, where there are well defined rows and columns.  This is the sort of data you would typically see in Microsoft Excel.  An example of tabular data is shown below.
# 
# Neural networks require numeric input.  This numeric form is called a feature vector.  Each row of training data typically becomes one vector.  The individual input neurons each receive one feature (or column) from this vector.  In this section, we will see how to encode the following tabular data into a feature vector.

# In[1]:


import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

print(df[0:5])


# The following observations can be made from the above data:
# * The target column is the column that you seek to predict.  There are several candidates here.  However, we will initially use product.  This field specifies what product someone bought.
# * There is an ID column.  This column should not be fed into the neural network as it contains no information useful for prediction.
# * Many of these fields are numeric and might not require any further processing.
# * The income column does have some missing values.
# * There are categorical values: job, area, and product.
# 
# To begin with, we will convert the job code into dummy variables.

# In[2]:


dummies = pd.get_dummies(df['job'],prefix="job")
print(dummies.shape)
print(dummies[0:10])


# Because there are 33 different job codes, there are 33 dummy variables.  We also specified a prefix, because the job codes (such as "ax") are not that meaningful by themselves.  Something such as "job_ax" also tells us the origin of this field.
# 
# Next, we must merge these dummies back into the main data frame.  We also drop the original "job" field, as it is now represented by the dummies. 

# In[3]:


df = pd.concat([df,dummies],axis=1)
df.drop('job', axis=1, inplace=True)
print(df[0:10])


# We also introduce dummy variables for the area column.

# In[4]:


df = pd.concat([df,pd.get_dummies(df['area'],prefix="area")],axis=1)
df.drop('area', axis=1, inplace=True)
print(df[0:10])


# The last remaining transformation is to fill in missing income values. 

# In[5]:


med = df['income'].median()
df['income'] = df['income'].fillna(med)


# There are more advanced ways of filling in missing values, but they require more analysis.  The idea would be to see if another field might give a hint as to what the income were.  For example, it might be beneficial to calculate a median income for each of the areas or job categories.  This is something to keep in mind for the class Kaggle competition.
# 
# At this point, the Pandas dataframe is ready to be converted to Numpy for neural network training. We need to know a list of the columns that will make up *x* (the predictors or inputs) and *y* (the target). 
# 
# The complete list of columns is:

# In[6]:


print(list(df.columns))


# This includes both the target and predictors.  We need a list with the target removed.  We also remove **id** because it is not useful for prediction.

# In[7]:


x_columns = df.columns.drop('product').drop('id')
print(list(x_columns))


# ### Generate X and Y for a Classification Neural Network

# We can now generate *x* and *y*.  Note, this is how we generate y for a classification problem.  Regression would not use dummies and would simply encode the numeric value of the target.

# In[8]:


# Convert to numpy - Classification
x_columns = df.columns.drop('product').drop('id')
x = df[x_columns].values
dummies = pd.get_dummies(df['product']) # Classification
products = dummies.columns
y = dummies.values


# We can display the *x* and *y* matrices.

# In[9]:


print(x)
print(y)


# The x and y values are now ready for a neural network.  Make sure that you construct the neural network for a classification problem.  Specifically,
# 
# * Classification neural networks have an output neuron count equal to the number of classes.
# * Classification neural networks should use **categorical_crossentropy** and a **softmax** activation function on the output layer.

# ### Generate X and Y for a Regression Neural Network
# 
# For a regression neural network, the *x* values are generated the same.  However, *y* does not use dummies.  Make sure to replace **income** with your actual target.

# In[10]:


y = df['income'].values


# # Module 4 Assignment
# 
# You can find the first assignment here: [assignment 4](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class1.ipynb)

# In[ ]:




