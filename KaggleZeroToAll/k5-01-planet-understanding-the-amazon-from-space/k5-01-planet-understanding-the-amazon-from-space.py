#!/usr/bin/env python
# coding: utf-8

# # Planet: Understanding the Amazon from Space
# 
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

# ## 1. Problem/data description
# 
# Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively
# 
# Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. While considerable research has been devoted to tracking changes in forests, it typically depends on coarse-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.
# 
# In this competition, Planet and its Brazilian partner SCCON are challenging Kagglers to label satellite image chips with atmospheric conditions and various classes of land cover/land use.

# ## 2. Loading data

# In[1]:


import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tqdm import tqdm


# In[2]:


x_train = []
y_train = []
x_test = []

df_train = pd.read_csv('input/train_v2.csv')

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}


# In[3]:


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file, protocol=4)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


# In[4]:


for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('input/train-jpg/{}.jpg'.format(f))
    img = cv2.resize(img, dsize=(64, 64))
    x_train.append(img)

    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)


# In[5]:


x_train = np.array(x_train, np.float32) / 255.
y_train = np.array(y_train, np.uint8)

cache_data(x_train, './x_train.dump')
cache_data(y_train, './y_train.dump')


# In[6]:


x_train = restore_data('./x_train.dump')
y_train = restore_data('./y_train.dump')


# ## 3. Model

# In[7]:


class Model(object):
    def __init__(self, sess, lr=0.001):
        self.sess = sess
        self.lr = lr
        self.build_model()
        self.sess.run(tf.global_variables_initializer())
        
    def build_model(self):
        self.X = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, 17], dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)
      
        filt1_1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        filt1_2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
        
        filt2_1 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        filt2_2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
        
        filt3_1 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        filt3_2 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01))
        
        filt4_1 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01))
        filt4_2 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01))
        
        fc_W1 = tf.Variable(tf.random_normal([4*4*256, 512], stddev=0.01))
        fc_W2 = tf.Variable(tf.random_normal([512, 17], stddev=0.01))
        
        scale = tf.Variable(tf.ones([3]))
        beta = tf.Variable(tf.zeros([3]))
        batch_mean, batch_var = tf.nn.moments(self.X, [0])
        self.X_bn = tf.nn.batch_normalization(self.X, batch_mean, batch_var, beta, scale, 1e-3)

        self.conv1_1 = tf.nn.relu(tf.nn.conv2d(self.X_bn, filt1_1, strides=[1, 1, 1, 1], padding='SAME'))
        self.conv1_2 = tf.nn.relu(tf.nn.conv2d(self.conv1_1, filt1_2, strides=[1, 1, 1, 1], padding='SAME'))
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        self.conv2_1 = tf.nn.relu(tf.nn.conv2d(self.pool1, filt2_1, strides=[1, 1, 1, 1], padding='SAME'))
        self.conv2_2 = tf.nn.relu(tf.nn.conv2d(self.conv2_1, filt2_2, strides=[1, 1, 1, 1], padding='SAME'))
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        self.conv3_1 = tf.nn.relu(tf.nn.conv2d(self.pool2, filt3_1, strides=[1, 1, 1, 1], padding='SAME'))
        self.conv3_2 = tf.nn.relu(tf.nn.conv2d(self.conv3_1, filt3_2, strides=[1, 1, 1, 1], padding='SAME'))
        self.pool3 = tf.nn.max_pool(self.conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        self.conv3_1 = tf.nn.relu(tf.nn.conv2d(self.pool2, filt3_1, strides=[1, 1, 1, 1], padding='SAME'))
        self.conv3_2 = tf.nn.relu(tf.nn.conv2d(self.conv3_1, filt3_2, strides=[1, 1, 1, 1], padding='SAME'))
        self.pool3 = tf.nn.max_pool(self.conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
        self.conv4_1 = tf.nn.relu(tf.nn.conv2d(self.pool3, filt4_1, strides=[1, 1, 1, 1], padding='SAME'))
        self.conv4_2 = tf.nn.relu(tf.nn.conv2d(self.conv4_1, filt4_2, strides=[1, 1, 1, 1], padding='SAME'))
        self.pool4 = tf.nn.max_pool(self.conv4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.fc1 = tf.reshape(self.pool4, [-1, fc_W1.get_shape().as_list()[0]])       
        self.fc1 = tf.nn.relu(tf.matmul(self.fc1, fc_W1))
        self.fc1 = tf.nn.dropout(self.fc1, self.dropout)
        
        scale1 = tf.Variable(tf.ones([512]))
        beta1 = tf.Variable(tf.zeros([512]))
        batch_mean1, batch_var1 = tf.nn.moments(self.fc1, [0])
        self.fc1_bn = tf.nn.batch_normalization(self.fc1, batch_mean1, batch_var1, beta1, scale1, 1e-3)
        
        self.fc2 = tf.matmul(self.fc1_bn, fc_W2)
        self.pred = tf.nn.sigmoid(self.fc2)
        
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.Y), axis=1))
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
    
    def fit(self, X, Y, epochs=10, batch_size=128, dropout=0.5):
        for epoch in range(epochs):
            avg_cost = 0 
            
            for step in range(0, len(X), batch_size):
                batch_mask = np.random.choice(len(X), batch_size) # For dataset shuffle

                feed_dict = {self.X: X[batch_mask], 
                             self.Y: Y[batch_mask], self.dropout: dropout}

                _, c = self.sess.run([self.train, self.cost], feed_dict=feed_dict)
                avg_cost += c
            
            avg_cost = avg_cost / (len(X) / batch_size)    
            print("epoch: {} cost: {:.5f}".format(epoch, avg_cost))
        print("Optimization Finished!")
        
    def pred_data(self, X, batch_size=256):
        preds = []
        for step in tqdm(range(0, len(X), batch_size)):
            feed_dict = {self.X: X[step:step+batch_size], 
                         self.dropout: 1}

            pred_batch = self.sess.run(self.pred, feed_dict=feed_dict)
            
            for pred in pred_batch:
                preds.append(pred)
        
        return preds
        
    def save_model(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        
    def restore_model(self, path):
        saver = tf.train.Saver()
        
        ckpt = tf.train.get_checkpoint_state(path)

        if ckpt and ckpt.model_checkpoint_path:
            print ('load learning')
            saver.restore(self.sess, ckpt.model_checkpoint_path)


# In[8]:


model = Model(tf.Session())


# In[9]:


model.fit(x_train, y_train, epochs=10, batch_size=512, dropout=0.5)


# In[10]:


model.save_model('./model/model')


# ## 4. Results

# In[11]:


model.restore_model('./model/model')


# In[12]:


for f, tags in tqdm(df_test.values, miniters=1000):
    img = cv2.imread('input/test-jpg/{}.jpg'.format(f))
    img = cv2.resize(img, dsize=(64, 64))
    x_test.append(img)


# In[13]:


x_test = np.array(x_test, np.float32) / 255.
cache_data(x_test, './x_test.dump')


# In[14]:


x_test = restore_data('./x_test.dump')


# In[15]:


pred = np.array(model.pred_data(x_test))


# In[16]:


labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

pred = pd.DataFrame(pred, columns = labels)


# In[19]:


result = []
thres = [0.07, 0.17, 0.2, 0.04, 0.23, 0.33, 0.24, 0.22, 0.1, 0.19, 0.23, 0.24, 0.12, 0.14, 0.25, 0.26, 0.16]
for i in tqdm(range(pred.shape[0]), miniters=1000):
    a = pred.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    result.append(' '.join(list(a.index)))


# In[20]:


df_test = pd.read_csv('input/sample_submission_v2.csv')
df_test['tags'] = result
df_test


# In[22]:


df_test.to_csv('submission.csv', index=False)


# ## 5. Future work/exercises

# There is nothing as good as a discussion item to get an insight into this.   
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/35902  
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/35797  
