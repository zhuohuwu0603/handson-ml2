#!/usr/bin/env python
# coding: utf-8

# Credits: Forked from [deep-learning-keras-tensorflow](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio

# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[1]:


import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import keras 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation


# For this section we will use the Kaggle otto challenge.
# If you want to follow, Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data

# #### About the data

# The Otto Group is one of the world’s biggest e-commerce companies, A consistent analysis of the performance of products is crucial. However, due to diverse global infrastructure, many identical products get classified differently.
# For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories. 
# Each row corresponds to a single product. There are a total of 93 numerical features, which represent counts of different events. All features have been obfuscated and will not be defined any further.
# 
# https://www.kaggle.com/c/otto-group-product-classification-challenge/data

# In[2]:


def load_data(path, train=True):
    """Load data from a CSV File
    
    Parameters
    ----------
    path: str
        The path to the CSV file
        
    train: bool (default True)
        Decide whether or not data are *training data*.
        If True, some random shuffling is applied.
        
    Return
    ------
    X: numpy.ndarray 
        The data as a multi dimensional array of floats
    ids: numpy.ndarray
        A vector of ids for each sample
    """
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)  # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


# In[3]:


def preprocess_data(X, scaler=None):
    """Preprocess input data by standardise features 
    by removing the mean and scaling to unit variance"""
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    """Encode labels with values among 0 and `n-classes-1`"""
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


# In[5]:


print("Loading data...")
X, labels = load_data('train.csv', train=True)
X, scaler = preprocess_data(X)
Y, encoder = preprocess_labels(labels)


X_test, ids = load_data('test.csv', train=False)
X_test, ids = X_test[:1000], ids[:1000]

#Plotting the data
print(X_test[:1])

X_test, _ = preprocess_data(X_test, scaler)

nb_classes = Y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')


# Now lets create and train a logistic regression model.

# ---

# # Keras
# 
# ## Deep Learning library for Theano and TensorFlow

# Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
# ref: https://keras.io/

# ## Why this name, Keras?
# 
# Keras (κέρας) means _horn_ in Greek. It is a reference to a literary image from ancient Greek and Latin literature, first found in the _Odyssey_, where dream spirits (_Oneiroi_, singular _Oneiros_) are divided between those who deceive men with false visions, who arrive to Earth through a gate of ivory, and those who announce a future that will come to pass, who arrive through a gate of horn. It's a play on the words κέρας (horn) / κραίνω (fulfill), and ἐλέφας (ivory) / ἐλεφαίρομαι (deceive).
# 
# Keras was initially developed as part of the research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot Operating System).
# 
# >_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

# ## Hands On - Keras Logistic Regression
# 

# In[6]:


dims = X.shape[1]
print(dims, 'dims')
print("Building model...")

nb_classes = Y.shape[1]
print(nb_classes, 'classes')

model = Sequential()
model.add(Dense(nb_classes, input_shape=(dims,)))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(X, Y)


# Simplicity is pretty impressive right? :)

# Now lets understand:
# <pre>The core data structure of Keras is a <b>model</b>, a way to organize layers. The main type of model is the <b>Sequential</b> model, a linear stack of layers.</pre>
# 

# What we did here is stacking a Fully Connected (<b>Dense</b>) layer of trainable weights from the input to the output and an <b>Activation</b> layer on top of the weights layer.

# ##### Dense

# ```python
# from keras.layers.core import Dense
# 
# Dense(output_dim, init='glorot_uniform', activation='linear', 
#       weights=None, W_regularizer=None, b_regularizer=None,
#       activity_regularizer=None, W_constraint=None, 
#       b_constraint=None, bias=True, input_dim=None)
# ```

# ##### Activation

# ```python
# from keras.layers.core import Activation
# 
# Activation(activation)
# ```

# ##### Optimizer

# If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
# Here we used <b>SGD</b> (stochastic gradient descent) as an optimization algorithm for our trainable weights.  

# "Data Sciencing" this example a little bit more
# =====

# What we did here is nice, however in the real world it is not useable because of overfitting.
# Lets try and solve it with cross validation.

# ##### Overfitting

# In overfitting, a statistical model describes random error or noise instead of the underlying relationship. Overfitting occurs when a model is excessively complex, such as having too many parameters relative to the number of observations. 
# 
# A model that has been overfit has poor predictive performance, as it overreacts to minor fluctuations in the training data.

# 
# <img src ="imgs/overfitting.png">

# <pre>To avoid overfitting, we will first split out data to training set and test set and test out model on the test set.
# Next: we will use two of keras's callbacks <b>EarlyStopping</b> and <b>ModelCheckpoint</b></pre>

# In[13]:


X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)
model.fit(X, Y, validation_data = (X_test, Y_test), nb_epoch=20, 
          batch_size=128, verbose=True, validation_split=0.15, 
          callbacks=[best_model, early_stop]) 


# ## Multi-Layer Perceptron and Fully Connected

# So, how hard can it be to build a Multi-Layer percepton with keras?
# It is baiscly the same, just add more layers!

# In[ ]:


model = Sequential()
model.add(Dense(100, input_shape=(dims,)))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(X, Y)


# Your Turn!

# ## Hands On - Keras Fully Connected
# 

# Take couple of minutes and Try and optimize the number of layers and the number of parameters in the layers to get the best results. 

# In[ ]:


model = Sequential()
model.add(Dense(100, input_shape=(dims,)))

# ...
# ...
# Play with it! add as much layers as you want! try and get better results.

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(X, Y)


# Building a question answering system, an image classification model, a Neural Turing Machine, a word2vec embedder or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

# #### Theoretical Motivations for depth

# >Much has been studied about the depth of neural nets. Is has been proven mathematically[1] and empirically that convolutional neural network benifit from depth! 

# [1] - On the Expressive Power of Deep Learning: A Tensor Analysis - Cohen, et al 2015

# #### Theoretical Motivations for depth

# One much quoted theorem about neural network states that:

# >Universal approximation theorem states[1] that a feed-forward network with a single hidden layer containing a finite number of neurons (i.e., a multilayer perceptron), can approximate continuous functions on compact subsets of $\mathbb{R}^n$, under mild assumptions on the activation function. The theorem thus states that simple neural networks can represent a wide variety of interesting functions when given appropriate parameters; however, it does not touch upon the algorithmic learnability of those parameters.

# [1] - Approximation Capabilities of Multilayer Feedforward Networks - Kurt Hornik 1991
