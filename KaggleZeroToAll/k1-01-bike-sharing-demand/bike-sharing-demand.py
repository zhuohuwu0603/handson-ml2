#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Demand
# https://www.kaggle.com/c/bike-sharing-demand
# 
# Code based on https://www.kaggle.com/klepacz/titanic/tensor-flow

# ## 1. Problem/data description
# 
# "You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period."

# ## 2. Loading data
# 
# First execute data_download.sh to download CSV files:
# ```bash
# $ sh data_download.sh
# ``` 
# 
# train.csv:
# ```csv
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed,casual,registered,count
# 2011-01-01 00:00:00,1,0,0,1,9.84,14.395,81,0,3,13,16
# 2011-01-01 01:00:00,1,0,0,1,9.02,13.635,80,0,8,32,40
# ```
# 
# test.csv:
# ```csv
# datetime,season,holiday,workingday,weather,temp,atemp,humidity,windspeed
# 2011-01-20 00:00:00,1,0,1,1,10.66,11.365,56,26.0027
# 2011-01-20 01:00:00,1,0,1,1,10.66,13.635,56,0
# ```
# 
# sample_submission.csv:
# ```csv
# datetime,count
# 2011-01-20 00:00:00,0
# 2011-01-20 01:00:00,0
# 2011-01-20 02:00:00,0
# 2011-01-20 03:00:00,0
# ```
# 

# In[1]:


# code to downlaod and laod 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility


# In[2]:


# Normalize x data
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# In[3]:


def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
        data = data_df.values[:, 1:] # Ignore datetime
        labels = data_df["datetime"].values
    else:
        data_df = pd.read_csv("train.csv")
        data = data_df.values[:, 1:-3] # Ignore datetime, and count, casual,registered
        labels = data_df["count"].values
    
    print(data_df.head(n=1))
    return labels, data


# In[4]:


# Load data and min/max 
# TODO: clean up this code
y_train, x_train = load_file(0)
y_train -= 1 # They are 1-7. So let's make it to 0~6
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)
# Get train file
testIds, x_test = load_file(1)

print(x_train.shape, x_test.shape)

x_all = np.vstack((x_train, x_test))
print(x_all.shape)

x_min_max_all = MinMaxScaler(x_all)
x_train = x_min_max_all[:train_len]
x_test = x_min_max_all[train_len:]

print(x_train.shape, x_test.shape)


# ## 3. Model (WIP)
# Model implementation. It can be divided to several small sections.

# In[5]:


# Parameters
learning_rate = 0.1

# Network Parameters
n_input = x_train.shape[1]
n_classes = 1  # regression

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, [None, 1])  # 0 ~ 6

W = tf.Variable(tf.random_normal([n_input, n_classes]), name='weight')
b = tf.Variable(tf.random_normal([n_classes]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[6]:


training_epochs = 100
batch_size = 32
display_step = 10
step_size = (int)(x_train.shape[0]/batch_size)+1
print(step_size)

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over step_size
        for step in range(step_size):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_data,
                                                          Y: batch_labels})
            avg_cost += c / step_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost))
            
    print("Optimization Finished!")
    
    ## 4. Results (creating submission file)
    
    outputs = sess.run(hypothesis, feed_dict={X: x_test})
    submission = ['Id,Cover_Type']

    for id, p in zip(testIds, outputs):
        submission.append('{0},{1}'.format(id, int(p))) 

    submission = '\n'.join(submission)

    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)


# ## 4. Results (creating submission file)
# (See above)

# ## 5. Future work/exercises
# * Wide and deep
# * RNN?
# * batch norm
