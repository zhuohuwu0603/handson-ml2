#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression in TensorFlow
# 
# Credits: Forked from [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) by Aymeric Damien
# 
# ## Setup
# 
# Refer to the [setup instructions](http://nbviewer.ipython.org/github/donnemartin/data-science-ipython-notebooks/blob/master/deep-learning/tensor-flow-examples/Setup_TensorFlow.md)

# In[5]:


# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[6]:


import tensorflow as tf


# In[7]:


# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1


# In[8]:


# tf Graph Input
x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes


# In[9]:


# Create model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# In[10]:


# Construct model
activation = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


# In[11]:


# Minimize error using cross entropy
# Cross entropy
cost = -tf.reduce_sum(y*tf.log(activation)) 
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 


# In[12]:


# Initializing the variables
init = tf.initialize_all_variables()


# In[13]:


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

