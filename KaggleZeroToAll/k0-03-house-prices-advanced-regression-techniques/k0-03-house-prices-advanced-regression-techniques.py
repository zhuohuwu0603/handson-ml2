#!/usr/bin/env python
# coding: utf-8

# # House Prices: Advanced Regression Techniques
# 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# 
# ### Reference   
# https://www.kaggle.com/apapiu/regularized-linear-models,   
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

# ## 0. Prepare
# 
# First execute data_download.sh to download CSV files:
# ```bash
# $ bash data_download.sh
# ``` 

# In[1]:


# code to downlaod and load 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import skew

tf.set_random_seed(777)  # for reproducibility

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Data description
# check data_description.txt
# 
# ```sh
# $ vi data_description.txt
# ```

# In[2]:


def scatter(data, var):
    matplotlib.rcParams['figure.figsize'] = (12.0, 4.0)
    data = pd.concat([data['SalePrice'], data[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));


# In[3]:


data = pd.read_csv("train.csv")


# In[4]:


matplotlib.rcParams['figure.figsize'] = (16.0, 6.0)
prices = pd.DataFrame({"log(price + 1)":np.log1p(data["SalePrice"]), "price":data["SalePrice"]})
prices.hist()


# In[5]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
corrmat = data.corr()
sns.heatmap(corrmat, vmax=.8, square=True);


# In[6]:


scatter(data, 'OverallQual')


# In[7]:


scatter(data, 'GrLivArea')


# ## 2. Loading data

# In[8]:


def load_file(is_test=False):
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        
    data_df = pd.concat((train.loc[:, cols],
                         test.loc[:, cols]))

    numeric_feats = data_df.dtypes[data_df.dtypes != 'object'].index
    skewed_feats = data_df[numeric_feats].apply(lambda x: skew(x.dropna())) 
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    data_df[skewed_feats] = np.log1p(data_df[skewed_feats])
    data_df = pd.get_dummies(data_df)
    data_df = data_df.fillna(data_df.mean())
        
    return data_df, train, test


# In[9]:


all_data, train, test = load_file()
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y_train = np.expand_dims(np.log1p(train.SalePrice), axis=1)

print(x_train.shape, y_train.shape)


# In[10]:


all_data.head()


# ## 3. Model
# Model implementation. It can be divided to several small sections.

# In[11]:


# Parameters
learning_rate = 0.001

# Network Parameters
n_input = x_train.shape[1]

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([n_input, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis- Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# TODO: Accuracy computation


# In[12]:


training_epochs = 2001
step_size=50

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training cycle
for epoch in range(training_epochs):
    
    avg_cost = 0 
    for step in range(0, len(x_train), step_size):
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: x_train[step:step+step_size], 
                                                      Y: y_train[step:step+step_size]})
        avg_cost += c
    
    avg_cost = avg_cost / len(x_train)

    if epoch % 250 == 0 :
        print("epoch: {} cost: {:.4f}".format(epoch, avg_cost))
        
print("Optimization Finished!")


# ## 4. Results
# Show the result

# In[13]:


# inverse of the log1p is expm1
test_price = np.squeeze(np.expm1(sess.run(hypothesis, feed_dict={X: x_test})), axis=(1,)) 
train_price = np.squeeze(np.expm1(sess.run(hypothesis, feed_dict={X: x_train})), axis=(1,))


# In[14]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
price=train["SalePrice"]


# In[15]:


test_tmp = np.concatenate((test_price[:], test_price[0:1])) # for match shape

prices = pd.DataFrame({"test price": test_tmp, "train price":train_price, "real price":price})

for array in prices.hist(bins=50):
    for subplot in array:
        subplot.set_xlim((0, 800000))


# In[16]:


ids = pd.read_csv("test.csv").Id
submission = ['Id,SalePrice']

for id, prediction in zip(ids, test_price):
    submission.append('{0},{1}'.format(id, prediction))

submission = '\n'.join(submission)

with open('submission.csv', 'w') as outfile:
    outfile.write(submission)


# ## 5. Future work/exercises

# Feature adjustments, data refinement
