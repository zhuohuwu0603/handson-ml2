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
# * **Part 2.1: Introduction to Pandas** [[Video]](https://www.youtube.com/watch?v=bN4UuCBdpZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_1_python_pandas.ipynb)
# * Part 2.2: Categorical Values [[Video]](https://www.youtube.com/watch?v=4a1odDpG0Ho&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_2_pandas_cat.ipynb)
# * Part 2.3: Grouping, Sorting, and Shuffling in Python Pandas [[Video]](https://www.youtube.com/watch?v=YS4wm5gD8DM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_3_pandas_grouping.ipynb)
# * Part 2.4: Using Apply and Map in Pandas for Keras [[Video]](https://www.youtube.com/watch?v=XNCEZ4WaPBY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_4_pandas_functional.ipynb)
# * Part 2.5: Feature Engineering in Pandas for Deep Learning in Keras [[Video]](https://www.youtube.com/watch?v=BWPTj4_Mi9E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_5_pandas_features.ipynb)

# # Part 2.1: Introduction to Pandas
# 
# [Pandas](http://pandas.pydata.org/) is an open source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.  It is based on the [dataframe](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) concept found in the [R programming language](https://www.r-project.org/about.html).  For this class, Pandas will be the primary means by which data is manipulated in conjunction with neural networks.
# 
# The dataframe is a key component of Pandas.  We will use it to access the [auto-mpg dataset](https://archive.ics.uci.edu/ml/datasets/Auto+MPG).  This dataset can be found on the UCI machine learning repository.  For this class we will use a version of the Auto MPG dataset where I added column headers.  You can find my version [here](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/data/auto-mpg.csv).
# 
# This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition.  It contains data for 398 cars, including [mpg](https://en.wikipedia.org/wiki/Fuel_economy_in_automobiles), [cylinders](https://en.wikipedia.org/wiki/Cylinder_(engine)), [displacement](https://en.wikipedia.org/wiki/Engine_displacement), [horsepower](https://en.wikipedia.org/wiki/Horsepower) , weight, acceleration, model year, origin and the car's name.
# 
# The following code loads the MPG dataset into a dataframe:

# In[1]:


# Simple dataframe
import os
import pandas as pd

df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/auto-mpg.csv")
print(df[0:5])


# The **display** function provides a cleaner display than simply printing the dataframe.

# In[2]:


# print(df[0:5])
print(df[0:5])

# It is possible to generate a second dataframe to display statistical information about the first dataframe.

# In[3]:


# Strip non-numerics
df = df.select_dtypes(include=['int', 'float'])

headers = list(df.columns.values)
fields = []

for field in headers:
    fields.append({
        'name' : field,
        'mean': df[field].mean(),
        'var': df[field].var(),
        'sdev': df[field].std()
    })

for field in fields:
    print(field)


# Converting this to a dataframe allows for a nicer display.

# In[4]:


df2 = pd.DataFrame(fields)
# print(df2)
print(df2)

# ## Missing Values
# 
# Missing values are a reality of machine learning.  Ideally every row of data will have values for all columns.  However, this is rarely the case.  Most of the values are present in the MPG database.  However, there are missing values in the horsepower column.  A common practice is to replace missing values with the median value for that column.  The median is calculated as described [here](https://www.mathsisfun.com/median.html).  The following code replaces any NA values in horsepower with the median:

# In[5]:


import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])
print(f"horsepower has na? {pd.isnull(df['horsepower']).values.any()}")
    
print("Filling missing values...")
med = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(med)
# df = df.dropna() # you can also simply drop NA values
                 
print(f"horsepower has na? {pd.isnull(df['horsepower']).values.any()}")


# # Dealing with Outliers
# 
# Outliers are values that are unusually high or low.  Sometimes outliers are simply errors, this is a result of [observation error](https://en.wikipedia.org/wiki/Observational_error).  Outliers can also be truly large or small values that may be difficult to account for. Outliers are usually defined as a value that is several standard deviations from the mean.  The following function can be used to remove such values.    

# In[6]:


# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# The code below will drop every row from the Auto MPG dataset where the horsepower is more than 2 standard deviations above or below the mean.

# In[7]:


import pandas as pd
import os
import numpy as np
from sklearn import metrics
from scipy.stats import zscore

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

# create feature vector
med = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(med)

# Drop the name column
df.drop('name',1,inplace=True)

# Drop outliers in horsepower
print("Length before MPG outliers dropped: {}".format(len(df)))
remove_outliers(df,'mpg',2)
print("Length after MPG outliers dropped: {}".format(len(df)))

# print(df[0:5])
print(df[0:5])

# ## Dropping Fields
# 
# Some fields are of no value to the neural network and can be dropped.  The following code removes the name column from the MPG dataset.

# In[8]:


import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

print(f"Before drop: {list(df.columns)}")
df.drop('name', 1, inplace=True)
print(f"After drop: {list(df.columns)}")


# ## Concatenating Rows and Columns
# Rows and columns can be concatenated together to form new data frames.  The code below creates a new dataframe from name and horsepower from the Auto MPG dataset.  This is done by concatenating two columns together.

# In[9]:


# Create a new dataframe from name and horsepower

import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

col_horsepower = df['horsepower']
col_name = df['name']
result = pd.concat([col_name, col_horsepower], axis=1)
# print(result[0:5])
print(result[0:5])

# The **concat** function can also concatenate two rows together.  This code concatenates the first 2 rows and the last 2 rows of the Auto MPG dataset.

# In[10]:


# Create a new dataframe from first 2 rows and last 2 rows

import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

result = pd.concat([df[0:2],df[-2:]], axis=0)
# print(result)
print(result)

# ## Training and Validation
# 
# It is very important that we evaluate a machine learning model based on its ability to predict data that it has never seen before.  Because of this we often divide the training data into a validation and training set.  The machine learning model will learn from the training data, but ultimately be evaluated based on the validation data.
# 
# * **Training Data** - **In Sample Data** - The data that the machine learning model was fit to/created from. 
# * **Validation Data** - **Out of Sample Data** - The data that the machine learning model is evaluated upon after it is fit to the training data.
# 
# There are two predominant means of dealing with training and validation data:
# 
# * **Training/Validation Split** - The data are split according to some ratio between a training and validation (hold-out) set.  Common ratios are 80% training and 20% validation.
# * **K-Fold Cross Validation** - The data are split into a number of folds and models.  Because a number of models equal to the folds is created out-of-sample predictions can be generated for the entire dataset.
# 
# The code below performs a split of the MPG data into a training and validation set.  The training set uses 80% of the data and the validation set uses 20%.
# 
# The following image shows how a model is trained on 80% of the data and then validated against the remaining 20%.
# 
# ![Training and Validation](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_1_train_val.png "Training and Validation")
# 

# In[11]:


import os
import pandas as pd
import numpy as np

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

df = df.reindex(np.random.permutation(df.index)) # Usually a good idea to shuffle
mask = np.random.rand(len(df)) < 0.8
trainDF = pd.DataFrame(df[mask])
validationDF = pd.DataFrame(df[~mask])

print(f"Training DF: {len(trainDF)}")
print(f"Validation DF: {len(validationDF)}")


# ### Converting a Dataframe to a Matrix
# 
# Neural networks do not directly operate on Python dataframes.  A neural network requires a numeric matrix.  The **values** property of a dataframe is used to convert to a matrix.

# In[12]:


df.values


# You might wish to only convert some of the columns, to leave out the name column, use the following code.

# In[13]:


df[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']].values


# ## Saving a Dataframe to CSV
# 
# Many of the assignments in this course will require that you save a dataframe to submit to the instructor.  The following code performs a shuffle and then saves a new copy.

# In[14]:


import os
import pandas as pd
import numpy as np

path = "."

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

filename_write = os.path.join(path, "auto-mpg-shuffle.csv")
df = df.reindex(np.random.permutation(df.index))
df.to_csv(filename_write, index=False) # Specify index = false to not write row numbers
print("Done")


# ## Saving a Dataframe to Pickle
# 
# CSV files are text and can be used by a variety of software programs.  However, they do take longer to generate and can sometimes lose small amounts of precision in the conversion.  Another format is [Pickle](https://docs.python.org/3/library/pickle.html).  Generally you will output to CSV because it is very compatible, even outside of Python.  The code below stores the Dataframe to Pickle.

# In[15]:


import os
import pandas as pd
import numpy as np
import pickle

path = "."

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

filename_write = os.path.join(path, "auto-mpg-shuffle.pkl")
df = df.reindex(np.random.permutation(df.index))

with open(filename_write,"wb") as fp:
    pickle.dump(df, fp)

print("Done")


# Loading the pickle file back into memory is accomplished by the following lines of code.  Notice that the index numbers are still jumbled from the previous shuffle?  Loading the CSV rebuilt (in the previous step) did not preserve these values.

# In[16]:


import os
import pandas as pd
import numpy as np
import pickle

path = "."

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv",
    na_values=['NA','?'])

filename_read = os.path.join(path, "auto-mpg-shuffle.pkl")

with open(filename_write,"rb") as fp:
    df = pickle.load(fp)

# print(df[0:5])
print(df[0:5])

# # Module 2 Assignment
# 
# You can find the first assignment here: [assignment 2](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class2.ipynb)

# In[ ]:




