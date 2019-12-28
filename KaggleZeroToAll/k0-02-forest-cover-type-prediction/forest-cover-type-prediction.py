#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction
# https://www.kaggle.com/c/forest-cover-type-prediction
# 
# Code based on https://www.kaggle.com/klepacz/titanic/tensor-flow

# ## 1. Problem/data description
# 
# "Predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data)."

# ## 2. Loading data
# 
# First execute data_download.sh to download CSV files:
# ```bash
# $ sh data_download.sh
# ``` 
# 
# You will have `train.csv.zip' and CSV looks something like this:
# ```
# Id  Elevation  Aspect  Slope  Horizontal_Distance_To_Hydrology  \
# 0   1       2596      51      3                               25...
# ...
# [n rows x 56 columns]
# ```

# In[1]:


# code to downlaod and laod 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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
        data_df = pd.read_csv("test.csv.zip", compression='zip')
        data = data_df.values[:, 1:] # Ignore ID
        labels = data_df["Id"].values
    else:
        data_df = pd.read_csv("train.csv.zip", compression='zip')
        data = data_df.values[:, 1:-1] # Ignore ID and Cover_Type
        labels = data_df["Cover_Type"].values
    
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


# ## 3. Model
# Model implementation. It can be divided to several small sections.

# In[5]:


# Parameters
learning_rate = 0.1

# Network Parameters
n_input = x_train.shape[1]
n_classes = 7  # 0 ~ 6

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
Y_one_hot = tf.one_hot(Y, n_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, n_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([n_input, n_classes]), name='weight')
b = tf.Variable(tf.random_normal([n_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[6]:


training_epochs = 15
batch_size = 32
display_step = 1
step_size = 1000

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_accuracy = 0.
        # Loop over step_size
        for step in range(step_size):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = x_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X: batch_data,
                                                          Y: batch_labels})
            avg_cost += c / step_size
            avg_accuracy += a / step_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost), "train accuracy={:.4f}".format(avg_accuracy))
    print("Optimization Finished!")
    
    ## 4. Results (creating submission file)
    
    outputs = sess.run(prediction, feed_dict={X: x_test})
    outputs += 1 # +1 to make 1-7
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
# * batch norm
