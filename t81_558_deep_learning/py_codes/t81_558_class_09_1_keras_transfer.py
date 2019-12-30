#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 9: Regularization: L1, L2 and Dropout**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 9 Material
# 
# * **Part 9.1: Introduction to Keras Transfer Learning** [[Video]](https://www.youtube.com/watch?v=WLlP6S-Z8Xs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_1_keras_transfer.ipynb)
# * Part 9.2: Popular Pretrained Neural Networks for Keras [[Video]](https://www.youtube.com/watch?v=ctVA1_46YEE&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_2_popular_transfer.ipynb)
# * Part 9.3: Transfer Learning for Computer Vision and Keras [[Video]](https://www.youtube.com/watch?v=61vMUm_XBMI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_3_transfer_cv.ipynb)
# * Part 9.4: Transfer Learning for Languages and Keras [[Video]](https://www.youtube.com/watch?v=ajmAAg9FxXA&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_4_transfer_nlp.ipynb)
# * Part 9.5: Transfer Learning for Keras Feature Engineering [[Video]](https://www.youtube.com/watch?v=Dttxsm8zpL8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_5_transfer_feature_eng.ipynb)

# # Part 9.1: Introduction to Keras Transfer Learning
# 
# Human beings learn new skills throughout their entire lives.  However, this learning is rarely from scratch.  No matter what task a human is learning, they are most likely drawing on experiences from earlier in life to learn this new skill.  In this way, humans learn much differently than most deep learning projects. 
# 
# At some point, a human being learns to tell the difference between a cat and a dog.  To teach a neural network the difference you would obtain a large quantity of cat pictures and dog pictures.  The neural network would iterate over all of these pictures and train on the differences.  The human child that learned to distinguish between the two animals would probably just need to see a few examples where they were told they were looking at each type of animal.  The human child would use previous knowledge of looking at different living and non-living objects to help make this classification.  The child would already know what sub-objects, such as fur, eyes, ears, noses, tails, and teeth looked like.
# 
# Transfer learning attempts to teach a neural network by similar means.  Rather than training your neural network from scratch, you begin training with preloaded set of weights. Usually you will simply remove the top-most layers of the pretrained neural network and retrain it with new topmost layers.  The layers remaining from the previous neural network will be locked so that training does not change these weights.  Only the newly added layers will be trained.  This process is summarized in the following figure.
# 
# ![Transfer Learning](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/transfer.png "Transfer Learning")
# 
# It can take a great deal of compute power to train a neural network for a large image dataset.  Google, Facebook, Microsoft, and other tech companies have utilized GPU arrays to train high quality neural networks for a variety of applications.  Transferring these weights into your neural network can save you considerable effort and compute time.  It is unlikely that a pretrained model will exactly fit the application that you seek to implement.  Finding the closest pretrained model and using transfer learning is an important skill for a deep learning engineer.
# 
# The [imagenet dataset]() has several high quality neural networks fit to it.  In the next parts of this module we will take a much closer look at this data set.  For many image recognition tasks, an imagenet trained neural network can be a great starting point for your own neural networks.
# 
# ### Transfer Learning Example
# 
# Lets look at an example of where transfer learning could be used to build upon an imagenet neural network.  Microsoft released a website that accepts a picture of a dog and attempts to classify these dogs by breed. The Microsoft dog breed website is provided here: 
# 
# [What breed is that dog?](https://www.bing.com/visualsearch/Microsoft/WhatDog)
# 
# To do this, it ie necessary to obtain pictures of dogs, labeled according to breed. Such a network could be trained entirely from scratch.  However, it would require a large quantity of breed-labeled pictures.  Transfer learning with imagenet could be very beneficial for a neural network project such as this.  A neural network pre-trained on imagenet would already contain neurons that are able to recognize many subcomponents of the various dog breeds that the neural network had previously seen on the other animal images in imagenet.
# 
# 
# 

# In[1]:


import pandas as pd
import io
import requests
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/iris.csv", 
    na_values=['NA', '?'])

# Convert to numpy - Classification
x = df[['sepal_l', 'sepal_w', 'petal_l', 'petal_w']].values
dummies = pd.get_dummies(df['species']) # Classification
species = dummies.columns
y = dummies.values


# Build neural network
model = Sequential()
model.add(Dense(50, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(25, activation='relu')) # Hidden 2
model.add(Dense(y.shape[1],activation='softmax')) # Output

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x,y,verbose=2,epochs=100)


# To keep this example simple, we are not setting aside a validation set.  The goal of this example is to show how to create a multi-layer neural network and transfer the weights to another.  Next we evaluate the accuracy on the training set.
# 

# In[2]:


from sklearn.metrics import accuracy_score
pred = model.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")


# Viewing the model summary is as expected, we can see the three layers previously defined.

# In[3]:


model.summary()


# Now that we've trained a neural network on the iris dataset the knowledge of this neural network can be transferred to other neural networks.  It is possible to create a new neural network from some or all of the layers of this neural network.  Just to demonstrate the technique, we will create a new neural network that is essentially a clone of the first neural network.  This is done by transferring all of the layers from the original neural network into the new one.

# In[4]:


model2 = Sequential()
for layer in model.layers:
    model2.add(layer)
model2.summary()


# As a sanity check, we would like to calculate the accuracy of the newly created model.  The in-sample accuracy should be the same as the previous model that the new model is based on.

# In[5]:


from sklearn.metrics import accuracy_score
pred = model2.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")


# The in-sample accuracy of the newly created neural network is the same as the original neural network.  We've successfully transferred all of the layers from the original neural network.
# 
# For this example we are going to train a neural network to classify three new hypothetical flowers that are uncreatively named:
# 
# * Fake Flower 1
# * Fake Flower 2
# * Fake Flower 3
# * Fake Flower 4
# 
# We have measurements for samples of these flowers that conform to the predictors contained in the original iris dataset: sepal width, sepal length, petal width, and petal length. For transfer learning to be effective the input for the newly trained neural network most closely conform to the original neural network that we are transferring from. 
# 
# We will strip away the last output layer that contains the softmax activation function that performs this final classification.  A new output layer will be created that will classify the 4 new flowers.  Only the weights in this new layer will be trained.  The first two layers will be marked as non-trainable.  The hope is that the first few layers have learned to abstract the raw input data in a way that is also helpful to the new neural network.
# 
# This is done by looping over the first few layers and copying them to the new neural network. We output a summary of the new neural network to verify that the previous output layer has been stripped.

# In[6]:


model3 = Sequential()
for i in range(2):
    layer = model.layers[i]
    layer.trainable = False
    model3.add(layer)
model3.summary()


# To complete the new neural network we add a 4-neuron classification layer and compile for classification.

# In[7]:


model3.add(Dense(4,activation='softmax')) # Output

model3.compile(loss='categorical_crossentropy', optimizer='adam')
model3.summary()


# Next we generate some training data for the 4 fake flowers, and train the neural network.

# In[8]:


x = np.array([
    [2.1,0.9,0.8,1.1], # 1
    [2.5,1.2,0.8,1.2],
    [1.1,3.1,1.1,1.1], # 2
    [0.8,2.2,0.7,1.2],
    [1.2,0.7,3.1,1.1], # 3
    [1.0,1.1,2.4,0.9],
    [0.1,1.1,4.1,1.2], # 4
    [1.2,0.8,3.1,0.1],
])

y = np.array([
    [0,0,0,1],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,1,0,0],
    [1,0,0,0],
    [1,0,0,0],
])

model3.fit(x,y,verbose=0,epochs=1000)


# We can evaluate the in-sample accuracy for the new model, that contains transferred layers from the previous model.

# In[9]:


from sklearn.metrics import accuracy_score
pred = model3.predict(x)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y,axis=1)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")


# # Module 9 Assignment
# 
# You can find the first assignment here: [assignment 9](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class9.ipynb)

# In[ ]:




