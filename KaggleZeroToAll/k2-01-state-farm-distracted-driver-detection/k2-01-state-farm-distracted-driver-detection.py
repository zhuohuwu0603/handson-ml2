#!/usr/bin/env python
# coding: utf-8

# # State Farm Distracted Driver Detection
# https://www.kaggle.com/c/state-farm-distracted-driver-detection
# 
# ## Reference
# https://www.kaggle.com/zfturbo/keras-sample

# ## 1. Problem/data description
# When you pass the offending driver, what do you expect to see? You certainly aren't surprised when you spot a driver who is texting, seemingly enraptured by social media, or in a lively hand-held conversation on their phone.
# 
# According to the CDC motor vehicle safety division, one in five car accidents is caused by a distracted driver. Sadly, this translates to 425,000 people injured and 3,000 people killed by distracted driving every year.
# 
# State Farm hopes to improve these alarming statistics, and better insure their customers, by testing whether dashboard cameras can automatically detect drivers engaging in distracted behaviors. Given a dataset of 2D dashboard camera images, State Farm is challenging Kagglers to classify each driver's behavior. Are they driving attentively, wearing their seatbelt, or taking a selfie with their friends in the backseat?  
# 
# ![](https://kaggle2.blob.core.windows.net/competitions/kaggle/5048/media/output_DEb8oT.gif)
#   
#   
# The 10 classes to predict are:  
# 
# c0: safe driving  
# c1: texting - right  
# c2: talking on the phone - right  
# c3: texting - left  
# c4: talking on the phone - left  
# c5: operating the radio  
# c6: drinking  
# c7: reaching behind  
# c8: hair and makeup  
# c9: talking to passenger  

# ## 2. Loading data
# First execute data_download.sh to download CSV/img files:
# 
# ```bash
# $ bash data_download.sh
# ```

# In[1]:


import os
import math
import time
import glob
import random
import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.set_random_seed(777)  # for reproducibility


# In[2]:


def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)
        
    # Image Rotation: make CNN Architecture cover rotating images.  
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    # Reduce size for reduce GPU memory usage and computation.
    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
    return resized


# In[3]:


def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    start_time = time.time()
    
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train', 'c' + str(j), '*.jpg')
        files = glob.glob(path) # Returns a list of all files and directories corresponding to the path.
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_train.append(img/255)
            y_train.append(j)
            
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    
    return X_train, y_train


# In[4]:


def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    start_time = time.time()
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img/255)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))
    
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    
    return X_test, X_test_id


# In[5]:


img_width = 64
img_height = 64


# In[6]:


x_train, y_train = load_train(img_width, img_height)
x_train = np.expand_dims(np.array(x_train, dtype=np.float32), axis=-1) 
y_train = np.array(y_train)

print(x_train.shape, y_train.shape)


# In[7]:


x_test, test_id = load_test(img_width, img_height)
x_test = np.expand_dims(np.array(x_test, dtype=np.float32), axis=-1) 
test_id = np.array(test_id)

print(x_test.shape, test_id.shape)


# ## 3. Model
# Model implementation. It can be divided to several small sections.

# In[8]:


# Hyperparameter
learning_rate = 0.0001
training_epochs = 10
batch_size = 32 # it is max batch size fit aws p2 instance GPU memory: 12G


# In[9]:


X = tf.placeholder(np.float32, shape=[None, img_width, img_height, 1])
Y = tf.placeholder(np.float32, shape=[None])
dropout = tf.placeholder(np.float32)

conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
pool1_dp = tf.nn.dropout(pool1, dropout)

conv2 = tf.layers.conv2d(inputs=pool1_dp, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_dp = tf.nn.dropout(pool2, dropout)

conv3 = tf.layers.conv2d(inputs=pool2_dp, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
pool3_dp = tf.nn.dropout(pool3, dropout)

pool3_flat = tf.reshape(pool3_dp, [-1, 8 * 8 * 128]) # w * h * d for pool3

fc1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
dropout1 = tf.nn.dropout(fc1, dropout)

fc2 = tf.layers.dense(inputs=dropout1, units=512, activation=tf.nn.relu)
dropout2 = tf.nn.dropout(fc2, dropout)

output = tf.layers.dense(inputs=dropout2, units=10)
pred = tf.nn.softmax(output)

onehot = tf.one_hot(indices=tf.cast(Y, tf.int32), depth=10)
cost = tf.losses.softmax_cross_entropy(onehot_labels=onehot, logits=output)
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(output, axis=1), tf.argmax(onehot, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    
    avg_cost = 0 
    avg_acc = 0
    
    for step in range(0, len(x_train), batch_size):
        batch_mask = np.random.choice(len(x_train), batch_size) # For dataset shuffle
        
        feed_dict = {X: x_train[batch_mask], 
                     Y: y_train[batch_mask], dropout: 0.5}
        
        _, c, a = sess.run([train, cost, accuracy], feed_dict=feed_dict)
        avg_cost += c
        avg_acc += a
        
    avg_cost = avg_cost / (len(x_train) / batch_size)    
    avg_acc = avg_acc / (len(x_train) / batch_size)    
    print("epoch: {} cost: {:.5f} acc: {:.5f}".format(epoch, avg_cost, avg_acc))

print("Optimization Finished!")


# ## 4. Results
# Show the result

# In[11]:


def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


# In[12]:


pred_arr = []
for step in range(0, len(x_test), batch_size):
    feed_dict = {X: x_test[step:step+batch_size], dropout: 1}

    preds = sess.run(pred, feed_dict=feed_dict)
    
    for tmp in preds:
        pred_arr.append(tmp)      


# In[13]:


create_submission(pred_arr, test_id)


# ## 5. Future work/exercises

# * Multiple Initialization Techniques  
# * Study of image size  
# * batch normalization
# * data augmentation
# * cross-validation
