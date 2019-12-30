#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 11: Natural Language Processing and Speech Recognition**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 11 Material
# 
# * Part 11.1: Getting Started with Spacy in Python [[Video]](https://www.youtube.com/watch?v=A5BtU9vXzu8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_01_spacy.ipynb)
# * **Part 11.2: Word2Vec and Text Classification** [[Video]](https://www.youtube.com/watch?v=nWxtRlpObIs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_02_word2vec.ipynb)
# * Part 11.3: What are Embedding Layers in Keras [[Video]](https://www.youtube.com/watch?v=OuNH5kT-aD0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_03_embedding.ipynb)
# * Part 11.4: Natural Language Processing with Spacy and Keras [[Video]](https://www.youtube.com/watch?v=BKgwjhao5DU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_04_text_nlp.ipynb)
# * Part 11.5: Learning English from Scratch with Keras and TensorFlow [[Video]](https://www.youtube.com/watch?v=Y1khuuSjZzc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=58) [[Notebook]](t81_558_class_11_05_english_scratch.ipynb)

# # Part 11.2: Word2Vec and Text Classification
# 
# Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are positioned in the vector space such that words that share common contexts in the corpus are located in close proximity to one another in the space.
# 
# Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient estimation of word representations in vector space](https://arxiv.org/abs/1301.3781). arXiv preprint arXiv:1301.3781.
# 
# ![Word2Vec](https://pbs.twimg.com/media/C7jJxIjWkAA8E_s.jpg)
# [Trust Word2Vec](https://twitter.com/DanilBaibak/status/844647217885581312)
# 
# ### Suggested Software for Word2Vec
# 
# * [GoogleNews Vectors](https://code.google.com/archive/p/word2vec/), [GitHub Mirror](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
# * [Python Gensim](https://radimrehurek.com/gensim/)
# 

# In[1]:


from tensorflow.keras.utils import get_file

try:
    path = get_file('GoogleNews-vectors-negative300.bin.gz', origin='https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
except:
    print('Error downloading')
    raise
    
print(path)    


# In[2]:


import gensim

# Not that the path below refers to a location on my hard drive.
# You should download GoogleNews Vectors (see suggested software above)
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)


# Word2vec makes each word a vector.  We are using the 300-number vector, which can be seen for the word "hello".

# In[3]:


w = model['hello']


# In[4]:


print(len(w))


# In[5]:


print(w)


# The code below shows the distance between two words.

# In[6]:


import numpy as np

w1 = model['king']
w2 = model['queen']

dist = np.linalg.norm(w1-w2)

print(dist)


# This shows the classic word2vec equation of **queen = (king - man) + female**

# In[7]:


model.most_similar(positive=['woman', 'king'], negative=['man'])


# The following code shows which item does not belong with the others.

# In[8]:


model.doesnt_match("house garage store dog".split())


# The following code shows the similarity between two words.

# In[9]:


model.similarity('iphone', 'android')


# The following code shows which words are most similar to the given one.

# In[10]:


model.most_similar('dog')


# In[ ]:




