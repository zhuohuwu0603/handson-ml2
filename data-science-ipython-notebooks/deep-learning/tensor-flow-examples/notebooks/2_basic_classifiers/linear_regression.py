#!/usr/bin/env python
# coding: utf-8

# # Linear Regression in TensorFlow
# 
# Credits: Forked from [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) by Aymeric Damien
# 
# ## Setup
# 
# Refer to the [setup instructions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/Setup_TensorFlow.md)

# In[2]:


import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random


# In[3]:


# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50


# In[4]:


# Training Data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


# In[5]:


# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")


# In[6]:


# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")


# In[7]:


# Construct a linear model
activation = tf.add(tf.mul(X, W), b)


# In[8]:


# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent


# In[9]:


# Initializing the variables
init = tf.initialize_all_variables()


# In[22]:


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=",                 "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})),                 "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    print "cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}),           "W=", sess.run(W), "b=", sess.run(b)

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


# In[23]:


from IPython.display import Image
Image(filename='linearreg.png')

