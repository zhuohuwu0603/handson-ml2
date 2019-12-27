#!/usr/bin/env python
# coding: utf-8

# Credits: Forked from [deep-learning-keras-tensorflow](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio

# Theano 
# ===
# A language in a language

# Dealing with weights matrices and gradients can be tricky and sometimes not trivial.
# Theano is a great framework for handling vectors, matrices and high dimensional tensor algebra. 
# Most of this tutorial will refer to Theano however TensorFlow is another great framework capable of providing an incredible abstraction for complex algebra.
# More on TensorFlow in the next chapters.

# In[1]:


import theano
import theano.tensor as T


# Symbolic variables
# ==========

# Theano has it's own variables and functions, defined the following

# In[2]:


x = T.scalar()


# In[ ]:


x


# Variables can be used in expressions

# In[4]:


y = 3*(x**2) + 1


# y is an expression now 

# Result is symbolic as well

# In[9]:


type(y)
y.shape


# #####printing

# As we are about to see, normal printing isn't the best when it comes to theano

# In[13]:


print(y)


# In[11]:


theano.pprint(y)


# In[24]:


theano.printing.debugprint(y)


# Evaluating expressions
# ============
# 
# Supply a `dict` mapping variables to values

# In[26]:


y.eval({x: 2})


# Or compile a function

# In[27]:


f = theano.function([x], y)


# In[28]:


f(2)


# Other tensor types
# ==========

# In[30]:


X = T.vector()
X = T.matrix()
X = T.tensor3()
X = T.tensor4()


# Automatic differention
# ============
# - Gradients are free!

# In[19]:


x = T.scalar()
y = T.log(x)


# In[20]:


gradient = T.grad(y, x)
print gradient
print gradient.eval({x: 2})
print (2 * gradient)


# # Shared Variables
# 
# - Symbolic + Storage

# In[39]:


import numpy as np
x = theano.shared(np.zeros((2, 3), dtype=theano.config.floatX))


# In[40]:


x


# We can get and set the variable's value

# In[41]:


values = x.get_value()
print(values.shape)
print(values)


# In[42]:


x.set_value(values)


# Shared variables can be used in expressions as well

# In[43]:


(x + 2) ** 2


# Their value is used as input when evaluating

# In[44]:


((x + 2) ** 2).eval()


# In[45]:


theano.function([], (x + 2) ** 2)()


# # Updates
# 
# - Store results of function evalution
# - `dict` mapping shared variables to new values

# In[46]:


count = theano.shared(0)
new_count = count + 1
updates = {count: new_count}

f = theano.function([], count, updates=updates)


# In[47]:


f()


# In[48]:


f()


# In[49]:


f()


# ### Warming up! Logistic Regression

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from keras.utils import np_utils


# For this section we will use the Kaggle otto challenge.
# If you want to follow, Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data

# #### About the data

# The Otto Group is one of the world’s biggest e-commerce companies, A consistent analysis of the performance of products is crucial. However, due to diverse global infrastructure, many identical products get classified differently.
# For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories. 
# Each row corresponds to a single product. There are a total of 93 numerical features, which represent counts of different events. All features have been obfuscated and will not be defined any further.
# 
# https://www.kaggle.com/c/otto-group-product-classification-challenge/data

# In[37]:


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


# In[38]:


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


# In[41]:


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

# #### Hands On - Logistic Regression

# In[46]:


#Based on example from DeepLearning.net
rng = np.random
N = 400
feats = 93
training_steps = 1

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)),
          allow_input_downcast=True)
predict = theano.function(inputs=[x], outputs=prediction, allow_input_downcast=True)

#Transform for class1
y_class1 = []
for i in Y:
    y_class1.append(i[0])
y_class1 = np.array(y_class1)

# Train
for i in range(training_steps):
    print('Epoch %s' % (i+1,))
    pred, err = train(X, y_class1)

print("target values for Data:")
print(y_class1)
print("prediction on training set:")
print(predict(X))


# In[ ]:




