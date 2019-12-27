#!/usr/bin/env python
# coding: utf-8

# # Nearest Neighbor in TensorFlow
# 
# Credits: Forked from [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) by Aymeric Damien
# 
# ## Setup
# 
# Refer to the [setup instructions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/Setup_TensorFlow.md)

# In[2]:


import numpy as np
import tensorflow as tf


# In[3]:


# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[4]:


# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing


# In[5]:


# Reshape images to 1D
Xtr = np.reshape(Xtr, newshape=(-1, 28*28))
Xte = np.reshape(Xte, newshape=(-1, 28*28))


# In[6]:


# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])


# In[8]:


# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)
# Predict: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.


# In[9]:


# Initializing the variables
init = tf.initialize_all_variables()


# In[10]:


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:]})
        # Get nearest neighbor class label and compare it to its true label
        print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]),               "True Class:", np.argmax(Yte[i])
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print "Done!"
    print "Accuracy:", accuracy


# In[ ]:




