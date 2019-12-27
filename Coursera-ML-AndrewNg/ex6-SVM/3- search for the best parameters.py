#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd
import scipy.io as sio


# # load data

# In[2]:


mat = sio.loadmat('./data/ex6data3.mat')
print(mat.keys())


# In[3]:


training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')


# In[4]:


print(training.shape)
training.head()


# In[5]:


print(cv.shape)
cv.head()


# # manual grid search for $C$ and $\sigma$
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

# In[6]:


candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


# In[7]:


# gamma to comply with sklearn parameter name
combination = [(C, gamma) for C in candidate for gamma in candidate]
len(combination)


# In[8]:


search = []

for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma)
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))


# In[9]:


best_score = search[np.argmax(search)]
best_param = combination[np.argmax(search)]

print(best_score, best_param)


# In[10]:


best_svc = svm.SVC(C=100, gamma=0.3)
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])

print(metrics.classification_report(cv['y'], ypred))


# # sklearn `GridSearchCV`
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

# In[11]:


parameters = {'C': candidate, 'gamma': candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)
clf.fit(training[['X1', 'X2']], training['y'])


# In[12]:


clf.best_params_


# In[13]:


clf.best_score_


# In[14]:


ypred = clf.predict(cv[['X1', 'X2']])
print(metrics.classification_report(cv['y'], ypred))


# >curiouly... they are not the same result. What?  
# 
# So the built in sklearn grid search is trying to find the best candidate from **training set**  
# However, when we were doing manual grid search, we train using training set, but we pick the best from **cross validation set**. This is the reason of difference.
# 
# ### I was wrong. That is not the reason
# It turns out that **GridSearch** will appropriate part of data as CV and use it to find the best candidate.  
# So the reason for different result is just that GridSearch here is just using part of **training data** to train because it need part of data as cv set

# In[ ]:




