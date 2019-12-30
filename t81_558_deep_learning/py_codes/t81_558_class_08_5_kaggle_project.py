#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 8: Kaggle Data Sets**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 8 Material
# 
# * Part 8.1: Introduction to Kaggle [[Video]](https://www.youtube.com/watch?v=v4lJBhdCuCU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_1_kaggle_intro.ipynb)
# * Part 8.2: Building Ensembles with Scikit-Learn and Keras [[Video]](https://www.youtube.com/watch?v=LQ-9ZRBLasw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_2_keras_ensembles.ipynb)
# * Part 8.3: How Should you Architect Your Keras Neural Network: Hyperparameters [[Video]](https://www.youtube.com/watch?v=1q9klwSoUQw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_3_keras_hyperparameters.ipynb)
# * Part 8.4: Bayesian Hyperparameter Optimization for Keras [[Video]](https://www.youtube.com/watch?v=sXdxyUCCm8s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_4_bayesian_hyperparameter_opt.ipynb)
# * **Part 8.5: Current Semester's Kaggle** [[Video]](https://www.youtube.com/watch?v=48OrNYYey5E) [[Notebook]](t81_558_class_08_5_kaggle_project.ipynb)
# 

# In[ ]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# # Part 8.5: Current Semester's Kaggle
# 
# Kaggke competition site for current semester (Fall 2019):
# 
# * [Fall 2019 Kaggle Assignment](https://kaggle.com/c/applications-of-deep-learningwustl-fall-2019)
# 
# Previous Kaggle competition sites for this class (NOT this semester's assignment, feel free to use code):
# * [Spring 2019 Kaggle Assignment](https://www.kaggle.com/c/applications-of-deep-learningwustl-spring-2019)
# * [Fall 2018 Kaggle Assignment](https://www.kaggle.com/c/wustl-t81-558-washu-deep-learning-fall-2018)
# * [Spring 2018 Kaggle Assignment](https://www.kaggle.com/c/wustl-t81-558-washu-deep-learning-spring-2018)
# * [Fall 2017 Kaggle Assignment](https://www.kaggle.com/c/wustl-t81-558-washu-deep-learning-fall-2017)
# * [Spring 2017 Kaggle Assignment](https://inclass.kaggle.com/c/applications-of-deep-learning-wustl-spring-2017)
# * [Fall 2016 Kaggle Assignment](https://inclass.kaggle.com/c/wustl-t81-558-washu-deep-learning-fall-2016)
# 

# # Iris as a Kaggle Competition
# 
# If the Iris data were used as a Kaggle, you would be given the following three files:
# 
# * [kaggle_iris_test.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_iris_test.csv) - The data that Kaggle will evaluate you on.  Contains only input, you must provide answers.  (contains x)
# * [kaggle_iris_train.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_iris_train.csv) - The data that you will use to train. (contains x and y)
# * [kaggle_iris_sample.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_iris_sample.csv) - A sample submission for Kaggle. (contains x and y)
# 
# Important features of the Kaggle iris files (that differ from how we've previously seen files):
# 
# * The iris species is already index encoded.
# * Your training data is in a separate file.
# * You will load the test data to generate a submission file.
# 
# The following program generates a submission file for "Iris Kaggle".  You can use it as a starting point for assignment 3.

# In[1]:


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

df_train = pd.read_csv("https://data.heatonresearch.com/data/t81-558/datasets/kaggle_iris_train.csv",
                       na_values=['NA','?'])

# Encode feature vector
df_train.drop('id', axis=1, inplace=True)

num_classes = len(df_train.groupby('species').species.nunique())

print("Number of classes: {}".format(num_classes))

# Convert to numpy - Classification
x = df_train[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
dummies = pd.get_dummies(df_train['species']) # Classification
species = dummies.columns
y = dummies.values
    
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.25, random_state=45)

# Train, with early stopping
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu'))
model.add(Dense(25))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
                       restore_best_weights=True)

model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)


# In[2]:


from sklearn import metrics

# Calculate multi log loss error
pred = model.predict(x_test)
score = metrics.log_loss(y_test, pred)
print("Log loss score: {}".format(score))


# In[3]:


# Generate Kaggle submit file

# Encode feature vector
df_test = pd.read_csv("https://data.heatonresearch.com/data/t81-558/datasets/kaggle_iris_test.csv",
                      na_values=['NA','?'])

# Convert to numpy - Classification
ids = df_test['id']
df_test.drop('id', axis=1, inplace=True)
x = df_test[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
y = dummies.values

# Generate predictions
pred = model.predict(x)
#pred

# Create submission data set

df_submit = pd.DataFrame(pred)
df_submit.insert(0,'id',ids)
df_submit.columns = ['id','species-0','species-1','species-2']

df_submit.to_csv("iris_submit.csv", index=False) # Write submit file locally

print(df_submit)


# ### MPG as a Kaggle Competition (Regression)
# 
# If the Auto MPG data were used as a Kaggle, you would be given the following three files:
# 
# * [kaggle_mpg_test.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_auto_test.csv) - The data that Kaggle will evaluate you on.  Contains only input, you must provide answers.  (contains x)
# * [kaggle_mpg_train.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_auto_test.csv) - The data that you will use to train. (contains x and y)
# * [kaggle_mpg_sample.csv](https://data.heatonresearch.com/data/t81-558/datasets/kaggle_auto_sample.csv) - A sample submission for Kaggle. (contains x and y)
# 
# Important features of the Kaggle iris files (that differ from how we've previously seen files):
# 
# The following program generates a submission file for "MPG Kaggle".  

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

save_path = "."

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/datasets/kaggle_auto_train.csv", 
    na_values=['NA', '?'])

cars = df['name']

# Handle missing value
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Pandas to Numpy
x = df[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']].values
y = df['mpg'].values # regression

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.25, random_state=42)

# Build the neural network
model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(10, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
                        verbose=1, mode='auto', restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),
          verbose=2,callbacks=[monitor],epochs=1000)

# Predict
pred = model.predict(x_test)


# In[5]:


import numpy as np

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))


# In[6]:


import pandas as pd

# Generate Kaggle submit file

# Encode feature vector
df_test = pd.read_csv("https://data.heatonresearch.com/data/t81-558/datasets/kaggle_auto_test.csv",
                      na_values=['NA','?'])

# Convert to numpy - regression
ids = df_test['id']
df_test.drop('id', axis=1, inplace=True)

# Handle missing value
df_test['horsepower'] = df_test['horsepower'].fillna(df['horsepower'].median())

x = df_test[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']].values




# Generate predictions
pred = model.predict(x)
#pred

# Create submission data set

df_submit = pd.DataFrame(pred)
df_submit.insert(0,'id',ids)
df_submit.columns = ['id','mpg']

df_submit.to_csv("auto_submit.csv", index=False) # Write submit file locally

print(df_submit)


# # Module 8 Assignment
# 
# You can find the first assignment here: [assignment 8](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class8.ipynb)

# In[ ]:




