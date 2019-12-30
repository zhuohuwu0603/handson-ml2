#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 13: Advanced/Other Topics**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 13 Video Material
# 
# * Part 13.1: Flask and Deep Learning Web Services [[Video]](https://www.youtube.com/watch?v=H73m9XvKHug&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_01_flask.ipynb)
# * Part 13.2: Deploying a Model to AWS  [[Video]](https://www.youtube.com/watch?v=8ygCyvRZ074&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_02_cloud.ipynb)
# * Part 13.3: Using a Keras Deep Neural Network with a Web Application  [[Video]](https://www.youtube.com/watch?v=OBbw0e-UroI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_03_web.ipynb)
# * **Part 13.4: When to Retrain Your Neural Network** [[Video]](https://www.youtube.com/watch?v=K2Tjdx_1v9g&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_04_retrain.ipynb)
# * Part 13.5: AI at the Edge: Using Keras on a Mobile Device  [[Video]]() [[Notebook]](t81_558_class_13_05_edge.ipynb)
# 

# # Part 13.4: When to Retrain Your Neural Network
# 
# * Dataset Shift
# * Covariate Shift
# 
# ![Covariate Shift](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/cov-shift.png "Covariate Shift")
# [(graph source)](http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf)
# 
# ### Measures of Drift
# 
# * Drift Detection Method (DDM)  [10], 
# * Early Drift Detection Method (EDDM)  [11], 
# * Page-Hinkley Test (PHT) [12], 
# * Adaptive Windowing (ADWIN)  [13], 
# * Paired Learners [14], 
# * EWMA for Concept Drift Detection (ECDD) [15], 
# * Degree of Drift (DOF) [16], and 
# * Statistical Test of Equal Proportions (STEPD) [17]
# 
# Others.
# 
# * KOLMOGOROV SMIRNOV TWO SAMPLE
# * https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
# 
# ### Academic/Other Sources
# 
# * [A unifying view on dataset shift in classification (paper)](https://rtg.cis.upenn.edu/cis700-2019/papers/dataset-shift/dataset-shift-terminology.pdf)
# * [A unifying view on dataset shift in classification (presentation)](http://reframe-d2k.org/img_auth.php/7/7e/Presentation_DatasetShift.pdf)
# * [A Comparative Study on Concept Drift Detectors](https://www.researchgate.net/publication/264081451_A_Comparative_Study_on_Concept_Drift_Detectors)
# * [Covariate Shift – Unearthing hidden problems in Real World Data Science](https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/)
# 
# 
# Kaggle data set:
# 
# * [Sberbank Russian Housing Market](https://www.kaggle.com/c/sberbank-russian-housing-market/data)

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PATH = "/Users/jheaton/Downloads/sberbank-russian-housing-market"


train_df = pd.read_csv(os.path.join(PATH,"train.csv"))
test_df = pd.read_csv(os.path.join(PATH,"test.csv"))


# In[2]:


def preprocess(df):
    for i in df.columns:
        if df[i].dtype == 'object':
            df[i] = df[i].fillna(df[i].mode().iloc[0])
        elif (df[i].dtype == 'int' or df[i].dtype == 'float'):
            df[i] = df[i].fillna(np.nanmedian(df[i]))

    enc = LabelEncoder()
    for i in df.columns:
        if (df[i].dtype == 'object'):
            df[i] = enc.fit_transform(df[i].astype('str'))
            df[i] = df[i].astype('object')


# In[3]:


preprocess(train_df)
preprocess(test_df)


# In[4]:


train_df.drop('price_doc',axis=1,inplace=True)


# ### KS-Statistic
# 
# We will use the KS-Statistic to look at the difference in distribution between columns in the training and test sets.  Just as a baseline, consider if we compare the same field to itself.  I this case we are comparing the **kitch_sq** in training set. Because there is no difference in distribution between a field in itself, the p-value is 1.0 and the KS-Statistic statistic is 0. The P-Value is the probability that there is no difference between two distributions.  Typically some lower threshold is used for how low of a P-Value is needed to reject the null hypothesis and assume there is a difference.  The value of 0.05 is a common threshold for p-values.  Because the p-value is NOT below 0.05 in this case we can assume the two distributions are the same.  If the p-value were below the threshold then the **statistic** value becomes interesting.  This value tells you how different the two distributions are.  A value of 0.0 in this case means no differences. 

# In[5]:


from scipy import stats

stats.ks_2samp(train_df['kitch_sq'], train_df['kitch_sq'])


# Now lets do something more interesting.  We will compare the same field **kitch_sq** between the test and training sets.  In this case, the p-value is below 0.05, so the **statistic** value now contains the amount of difference detected.

# In[6]:


stats.ks_2samp(train_df['kitch_sq'], test_df['kitch_sq'])


# Next we pull the KS-Stat for every field.  We also establish a boundary of what is the maximum p-value to display and how much of a difference is needed before we display the column.

# In[7]:


for col in train_df.columns:
    ks = stats.ks_2samp(train_df[col], test_df[col])
    if ks.pvalue < 0.05 and ks.statistic>0.1:
        print(f'{col}: {ks}')


# ### Detecting Drift between Training and Testing Datasets by Training

# Sample the training and test into smaller sets to train.  We would like 10K elements from each; however, the test set only has 7,662, so as a result we only sample that amount from each side.

# In[8]:


SAMPLE_SIZE = min(len(train_df),len(test_df))
SAMPLE_SIZE = min(SAMPLE_SIZE,10000)
print(SAMPLE_SIZE)


# We take the random samples from the training and test sets and also add a flag called **source_training** so we can tell the two apart.

# In[9]:


training_sample = train_df.sample(SAMPLE_SIZE, random_state=49)
testing_sample = test_df.sample(SAMPLE_SIZE, random_state=48)

# Is the data from the training set?
training_sample['source_training'] = 1
testing_sample['source_training'] = 0


# Next we combine the data that we sampled from the training and test data sets and shuffle the results.

# In[13]:


# Build combined training set
combined = testing_sample.append(training_sample)
combined.reset_index(inplace=True, drop=True)

# Now randomize
combined = combined.reindex(np.random.permutation(combined.index))
combined.reset_index(inplace=True, drop=True)


# We will now generate $x$ and $y$ to train.  We are attempting to predict the **source_training** value as $y$, that indicates if the data came from train or test.  If the model is very successful at using the data to predict if it came from train or test then there is likely drift.  Ideally the train and test data should be indistinguishable.  

# In[14]:


# Get ready to train
y = combined['source_training'].values
combined.drop('source_training',axis=1,inplace=True)
x = combined.values


# In[12]:


y


# We will consider anything above a 0.75 AUC as having a good chance of drift.

# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators = 60, max_depth = 7,min_samples_leaf = 5)
lst = []

for i in combined.columns:
    score = cross_val_score(model,pd.DataFrame(combined[i]),y,cv=2,scoring='roc_auc')
    if (np.mean(score) > 0.75):
        lst.append(i)
        print(i,np.mean(score))


# In[ ]:




