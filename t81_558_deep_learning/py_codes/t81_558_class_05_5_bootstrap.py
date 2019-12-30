#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 5: Regularization and Dropout**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 5 Material
# 
# * Part 5.1: Part 5.1: Introduction to Regularization: Ridge and Lasso [[Video]](https://www.youtube.com/watch?v=jfgRtCYjoBs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_05_1_reg_ridge_lasso.ipynb)
# * Part 5.2: Using K-Fold Cross Validation with Keras [[Video]](https://www.youtube.com/watch?v=maiQf8ray_s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_05_2_kfold.ipynb)
# * Part 5.3: Using L1 and L2 Regularization with Keras to Decrease Overfitting [[Video]](https://www.youtube.com/watch?v=JEWzWv1fBFQ&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_05_3_keras_l1_l2.ipynb)
# * Part 5.4: Drop Out for Keras to Decrease Overfitting [[Video]](https://www.youtube.com/watch?v=bRyOi0L6Rs8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_05_4_dropout.ipynb)
# * **Part 5.5: Benchmarking Keras Deep Learning Regularization Techniques** [[Video]](https://www.youtube.com/watch?v=1NLBwPumUAs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_05_5_bootstrap.ipynb)
# 

# # Part 5.5: Benchmarking Keras Deep Learning Regularization Techniques
# 
# Quite a few hyperparameters have been introduced so far.  Tweaking each of these values can have an effect on the score obtained by your neural networks.  Some of the hyperparameters seen so far include:
# 
# * Number of layers in the neural network
# * How many neurons in each layer
# * What activation functions to use on each layer
# * Dropout percent on each layer
# * L1 and L2 values on each layer
# 
# To try out each of these hyperparameters you will need to run train neural networks with multiple settings for each hyperparameter.  However, you may have noticed that neural networks often produce somewhat different results when trained multiple times.  This is because the neural networks start with random weights.  Because of this it is necessary to fit and evaluate a neural network times to ensure that one set of hyperparameters are actually better than another.  Bootstrapping can be an effective means of benchmarking (comparing) two sets of hyperparameters.  
# 
# Bootstrapping is similar to cross-validation.  Both go through a number of cycles/folds providing validation and training sets.  However, bootstrapping can have an unlimited number of cycles.  Bootstrapping chooses a new train and validation split each cycle, with replacement.  The fact that each cycle is chosen with replacement means that, unlike cross validation, there will often be repeated rows selected between cycles.  If you run the bootstrap for enough cycles, there will be duplicate cycles.
# 
# In this part we will use bootstrapping for hyperparameter benchmarking.  We will train a neural network for a specified number of splits (denoted by the SPLITS constant).  For these examples we use 100.  We will compare the average score at the end of the 100.  By the end of the cycles the mean score will have converged somewhat.  This ending score will be a much better basis of comparison than a single cross-validation.  Additionally, the average number of epochs will be tracked to give an idea of a possible optimal value.  Because the early stopping validation set is also used to evaluate the the neural network as well, it might be slightly inflated.  This is because we are both stopping and evaluating on the same sample.  However, we are using the scores only as relative measures to determine the superiority of one set of hyperparameters to another, so this slight inflation should not present too much of a problem.
# 
# Because we are benchmarking, we will display the amount of time taken for each cycle.  The following function can be used to nicely format a time span.

# In[1]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# ### Additional Reading on Hyperparameter Tuning
# 
# I will add more here as I encounter additional good sources:
#     
# * [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)

# ### Bootstrapping for Regression
# 
# Regression bootstrapping uses the **ShuffleSplit** object to perform the splits.  This is similar to **KFold** for cross validation, no balancing takes place.  To demonstrate this technique we will attempt to predict the age column for the jh-simple-dataset this data is loaded by the following code.

# In[2]:


import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# Read the data set
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

# Generate dummies for job
df = pd.concat([df,pd.get_dummies(df['job'],prefix="job")],axis=1)
df.drop('job', axis=1, inplace=True)

# Generate dummies for area
df = pd.concat([df,pd.get_dummies(df['area'],prefix="area")],axis=1)
df.drop('area', axis=1, inplace=True)

# Generate dummies for product
df = pd.concat([df,pd.get_dummies(df['product'],prefix="product")],axis=1)
df.drop('product', axis=1, inplace=True)

# Missing values for income
med = df['income'].median()
df['income'] = df['income'].fillna(med)

# Standardize ranges
df['income'] = zscore(df['income'])
df['aspect'] = zscore(df['aspect'])
df['save_rate'] = zscore(df['save_rate'])
df['subscriptions'] = zscore(df['subscriptions'])

# Convert to numpy - Classification
x_columns = df.columns.drop('age').drop('id')
x = df[x_columns].values
y = df['age'].values


# The following code performs the bootstrap.  The architecture of the neural network can be adjusted to compare many different configurations. 

# In[3]:


import pandas as pd
import os
import numpy as np
import time
import statistics
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ShuffleSplit

SPLITS = 50

# Bootstrap
boot = ShuffleSplit(n_splits=SPLITS, test_size=0.1, random_state=42)

# Track progress
mean_benchmark = []
epochs_needed = []
num = 0

# Loop through samples
for train, test in boot.split(x):
    start_time = time.time()
    num+=1

    # Split train and test
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    # Construct neural network
    model = Sequential()
    model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=5, verbose=0, mode='auto', restore_best_weights=True)

    # Train on the bootstrap sample
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)
    epochs = monitor.stopped_epoch
    epochs_needed.append(epochs)
    
    # Predict on the out of boot (validation)
    pred = model.predict(x_test)
  
    # Measure this bootstrap's log loss
    score = np.sqrt(metrics.mean_squared_error(pred,y_test))
    mean_benchmark.append(score)
    m1 = statistics.mean(mean_benchmark)
    m2 = statistics.mean(epochs_needed)
    mdev = statistics.pstdev(mean_benchmark)
    
    # Record this iteration
    time_took = time.time() - start_time
    print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")


# The bootstrapping process for classification is similar and is presented in the next section.

# ### Bootstrapping for Classification
# 
# Regression bootstrapping uses the **StratifiedShuffleSplit** object to perform the splits.  This is similar to **StratifiedKFold** for cross validation, as the classes are balanced so that the sampling has no effect on proportions.  To demonstrate this technique we will attempt to predict the product column for the jh-simple-dataset this data is loaded by the following code.

# In[4]:


import pandas as pd
from scipy.stats import zscore

# Read the data set
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

# Generate dummies for job
df = pd.concat([df,pd.get_dummies(df['job'],prefix="job")],axis=1)
df.drop('job', axis=1, inplace=True)

# Generate dummies for area
df = pd.concat([df,pd.get_dummies(df['area'],prefix="area")],axis=1)
df.drop('area', axis=1, inplace=True)

# Missing values for income
med = df['income'].median()
df['income'] = df['income'].fillna(med)

# Standardize ranges
df['income'] = zscore(df['income'])
df['aspect'] = zscore(df['aspect'])
df['save_rate'] = zscore(df['save_rate'])
df['age'] = zscore(df['age'])
df['subscriptions'] = zscore(df['subscriptions'])

# Convert to numpy - Classification
x_columns = df.columns.drop('product').drop('id')
x = df[x_columns].values
dummies = pd.get_dummies(df['product']) # Classification
products = dummies.columns
y = dummies.values


# In[5]:


import pandas as pd
import os
import numpy as np
import time
import statistics
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit

SPLITS = 50

# Bootstrap
boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1, random_state=42)

# Track progress
mean_benchmark = []
epochs_needed = []
num = 0

# Loop through samples
for train, test in boot.split(x,df['product']):
    start_time = time.time()
    num+=1

    # Split train and test
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    # Construct neural network
    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu')) # Hidden 1
    model.add(Dense(25, activation='relu')) # Hidden 2
    model.add(Dense(y.shape[1],activation='softmax')) # Output
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=25, verbose=0, mode='auto', restore_best_weights=True)

    # Train on the bootstrap sample
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)
    epochs = monitor.stopped_epoch
    epochs_needed.append(epochs)
    
    # Predict on the out of boot (validation)
    pred = model.predict(x_test)
  
    # Measure this bootstrap's log loss
    y_compare = np.argmax(y_test,axis=1) # For log loss calculation
    score = metrics.log_loss(y_compare, pred)
    mean_benchmark.append(score)
    m1 = statistics.mean(mean_benchmark)
    m2 = statistics.mean(epochs_needed)
    mdev = statistics.pstdev(mean_benchmark)
    
    # Record this iteration
    time_took = time.time() - start_time
    print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")


# ### Benchmarking
# 
# Now that we've seen how to bootstrap with both classification and regression we can start to try to optimize the hyperparameters for the jh-simple-dataset data.  For this example we will encode for classification of the product column.  Evaluation will be in log loss.

# In[6]:


import pandas as pd
from scipy.stats import zscore

# Read the data set
df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
    na_values=['NA','?'])

# Generate dummies for job
df = pd.concat([df,pd.get_dummies(df['job'],prefix="job")],axis=1)
df.drop('job', axis=1, inplace=True)

# Generate dummies for area
df = pd.concat([df,pd.get_dummies(df['area'],prefix="area")],axis=1)
df.drop('area', axis=1, inplace=True)

# Missing values for income
med = df['income'].median()
df['income'] = df['income'].fillna(med)

# Standardize ranges
df['income'] = zscore(df['income'])
df['aspect'] = zscore(df['aspect'])
df['save_rate'] = zscore(df['save_rate'])
df['age'] = zscore(df['age'])
df['subscriptions'] = zscore(df['subscriptions'])

# Convert to numpy - Classification
x_columns = df.columns.drop('product').drop('id')
x = df[x_columns].values
dummies = pd.get_dummies(df['product']) # Classification
products = dummies.columns
y = dummies.values


# I performed some optimization and the code is currently set to the best settings that I came up with. Later in this course we will see how we can use an automatic process to optimize the hyperparameters.

# In[7]:


import pandas as pd
import os
import numpy as np
import time
import tensorflow.keras.initializers
import statistics
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import LeakyReLU,PReLU

SPLITS = 100

# Bootstrap
boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1)

# Track progress
mean_benchmark = []
epochs_needed = []
num = 0

# Loop through samples
for train, test in boot.split(x,df['product']):
    start_time = time.time()
    num+=1

    # Split train and test
    x_train = x[train]
    y_train = y[train]
    x_test = x[test]
    y_test = y[test]

    # Construct neural network
    # kernel_initializer = tensorflow.keras.initializers.he_uniform(seed=None)
    model = Sequential()
    model.add(Dense(100, input_dim=x.shape[1], activation=PReLU(), kernel_regularizer=regularizers.l2(1e-4)
    )) # Hidden 1
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=PReLU(), activity_regularizer=regularizers.l2(1e-4)
    )) # Hidden 2
    model.add(Dropout(0.5))
    model.add(Dense(100, activation=PReLU(), activity_regularizer=regularizers.l2(1e-4)
    )) # Hidden 3
#    model.add(Dropout(0.5)) - Usually better performance without dropout on final layer
    model.add(Dense(y.shape[1],activation='softmax')) # Output
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=100, verbose=0, mode='auto', restore_best_weights=True)

    # Train on the bootstrap sample
    model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=0,epochs=1000)
    epochs = monitor.stopped_epoch
    epochs_needed.append(epochs)
    
    # Predict on the out of boot (validation)
    pred = model.predict(x_test)
  
    # Measure this bootstrap's log loss
    y_compare = np.argmax(y_test,axis=1) # For log loss calculation
    score = metrics.log_loss(y_compare, pred)
    mean_benchmark.append(score)
    m1 = statistics.mean(mean_benchmark)
    m2 = statistics.mean(epochs_needed)
    mdev = statistics.pstdev(mean_benchmark)
    
    # Record this iteration
    time_took = time.time() - start_time
    print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")
# https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404

# 100 Prelu 0.5 1-2, He init
# Bootstrap #100: Log Loss score=0.604399, mean score=0.651921, stdev=0.045944 epochs=298, time=0:00:40.12
# 100 Prelu 0.5 1-2
# Bootstrap #53: Log Loss score=0.652944, mean score=0.655620, stdev=0.050328 epochs=269, time=0:00:51.02


# In[ ]:




