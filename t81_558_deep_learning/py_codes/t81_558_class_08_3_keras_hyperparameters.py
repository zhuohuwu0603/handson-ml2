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
# * **Part 8.3: How Should you Architect Your Keras Neural Network: Hyperparameters** [[Video]](https://www.youtube.com/watch?v=1q9klwSoUQw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_3_keras_hyperparameters.ipynb)
# * Part 8.4: Bayesian Hyperparameter Optimization for Keras [[Video]](https://www.youtube.com/watch?v=sXdxyUCCm8s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_4_bayesian_hyperparameter_opt.ipynb)
# * Part 8.5: Current Semester's Kaggle [[Video]](https://www.youtube.com/watch?v=48OrNYYey5E) [[Notebook]](t81_558_class_08_5_kaggle_project.ipynb)
# 

# In[1]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# # Part 8.3: How Should you Architect Your Keras Neural Network: Hyperparameters
# 
# * [Guide to choosing Hyperparameters for your Neural Networks](https://towardsdatascience.com/guide-to-choosing-hyperparameters-for-your-neural-networks-38244e87dafe)
# 
# ### Number of Hidden Layers and Neuron Counts
# 
# * [Keras Layers](https://keras.io/layers/core/)
# 
# Layer types and when to use them:
# 
# * **Activation** - Layer that simply adds an activation function, the activation function can also be specified as part of a Dense (or other) layer type.
# * **ActivityRegularization** Used to add L1/L2 regularization outside of a layer.  L1 and L2 can also be specified as part of a Dense (or other) layer type.
# * **Dense** - The original neural network layer type.  Every neuron is connected to the next layer.  The input vector is one-dimensional and placing certain inputs next to each other does not have an effect. 
# * **Dropout** - Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.  Dropout only occurs during training.
# * **Flatten** - Flattens the input to 1D. Does not affect the batch size.
# * **Input** - A Keras tensor is a tensor object from the underlying backend (Theano, TensorFlow or CNTK), which we augment with certain attributes that allow us to build a Keras model just by knowing the inputs and outputs of the model.
# * **Lambda** - Wraps arbitrary expression as a Layer object.
# * **Masking** - Masks a sequence by using a mask value to skip timesteps.
# * **Permute** - Permutes the dimensions of the input according to a given pattern. Useful for e.g. connecting RNNs and convnets together.
# * **RepeatVector** - Repeats the input n times.
# * **Reshape** - Similar to Numpy reshapes.
# * **SpatialDropout1D** - This version performs the same function as Dropout, however it drops entire 1D feature maps instead of individual elements. 
# * **SpatialDropout2D** - This version performs the same function as Dropout, however it drops entire 2D feature maps instead of individual elements
# * **SpatialDropout3D** - This version performs the same function as Dropout, however it drops entire 3D feature maps instead of individual elements. 
# 
# 
# ### Activation Functions
# 
# * [Keras Activation Functions](https://keras.io/activations/)
# * [Activation Function Cheat Sheets](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)
# 
# As follows:
# 
# * **softmax** - Used for multi-class classification.  Ensures all output neurons behave as probabilities and sum to 1.0.
# * **elu** - Exponential linear unit.  Exponential Linear Unit or its widely known name ELU is a function that tend to converge cost to zero faster and produce more accurate results.  Can produce negative outputs.
# * **selu** - Scaled Exponential Linear Unit (SELU), essentially **elu** multiplied by a scaling constant.
# * **softplus** - Softplus activation function. $log(exp(x) + 1)$  [Introduced](https://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf) in 2001.
# * **softsign** Softsign activation function. $x / (abs(x) + 1)$ Similar to tanh, but not widely used.
# * **relu** - Very popular neural network activation function.  Used for hidden layers, cannot output negative values.  No trainable parameters.
# * **tanh** Classic neural network activation function, though often replaced by relu family on modern networks.
# * **sigmoid** - Classic neural network activation.  Often used on output layer of a binary classifier.
# * **hard_sigmoid** - Less computationally expensive variant of sigmoid.
# * **exponential** - Exponential (base e) activation function.
# * **linear** - Pass through activation function. Usually used on the output layer of a regression neural network.
# 
# ### Advanced Activation Functions
# 
# * [Keras Advanced Activation Functions](https://keras.io/layers/advanced-activations/)
# 
# The advanced activation functions contain parameters that are trained during neural network fitting. As follows:
# 
# * **LeakyReLU** - Leaky version of a Rectified Linear Unit. It allows a small gradient when the unit is not active, controlled by alpha hyperparameter.
# * **PReLU** - Parametric Rectified Linear Unit, learns the alpha hyperparameter. 
# 
# ### Regularization: L1, L2, Dropout
# 
# * [Keras Regularization](https://keras.io/regularizers/)
# * [Keras Dropout](https://keras.io/layers/core/)
# 
# ### Batch Normalization
# 
# * [Keras Batch Normalization](https://keras.io/layers/normalization/)
# 
# * Ioffe, S., & Szegedy, C. (2015). [Batch normalization: Accelerating deep network training by reducing internal covariate shift](https://arxiv.org/abs/1502.03167). *arXiv preprint arXiv:1502.03167*.
# 
# Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.  Can allow learning rate to be larger.
# 
# 
# ### Training Parameters
# 
# * [Keras Optimizers](https://keras.io/optimizers/)
# 
# * **Batch Size** - Usually small, such as 32 or so.
# * **Learning Rate**  - Usually small, 1e-3 or so.

# In[2]:


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


# In[3]:


import pandas as pd
import os
import numpy as np
import time
import tensorflow.keras.initializers
import statistics
import tensorflow.keras
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import LeakyReLU,PReLU
from tensorflow.keras.optimizers import Adam



def evaluate_network(dropout,lr,neuronPct,neuronShrink):
    SPLITS = 2

    # Bootstrap
    boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.1)

    # Track progress
    mean_benchmark = []
    epochs_needed = []
    num = 0
    neuronCount = int(neuronPct * 5000)

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
        
        layer = 0
        while neuronCount>25 and layer<10:
            #print(neuronCount)
            if layer==0:
                model.add(Dense(neuronCount, 
                    input_dim=x.shape[1], 
                    activation=PReLU())) 
            else:
                model.add(Dense(neuronCount, activation=PReLU())) 
            model.add(Dropout(dropout))
        
            neuronCount = neuronCount * neuronShrink
        
        model.add(Dense(y.shape[1],activation='softmax')) # Output
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
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
        #print(f"#{num}: score={score:.6f}, mean score={m1:.6f}, stdev={mdev:.6f}, epochs={epochs}, mean epochs={int(m2)}, time={hms_string(time_took)}")

    tensorflow.keras.backend.clear_session()
    return (-m1)

print(evaluate_network(
    dropout=0.2,
    lr=1e-3,
    neuronPct=0.2,
    neuronShrink=0.2))


# In[ ]:




