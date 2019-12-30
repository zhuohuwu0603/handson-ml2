#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 11: Natural Language Processing and Speech Recognition**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 11 Material
# 
# * Part 11.1: Getting Started with Spacy in Python [[Video]](https://www.youtube.com/watch?v=A5BtU9vXzu8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_01_spacy.ipynb)
# * Part 11.2: Word2Vec and Text Classification [[Video]](https://www.youtube.com/watch?v=nWxtRlpObIs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_02_word2vec.ipynb)
# * **Part 11.3: What are Embedding Layers in Keras** [[Video]](https://www.youtube.com/watch?v=OuNH5kT-aD0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_03_embedding.ipynb)
# * Part 11.4: Natural Language Processing with Spacy and Keras [[Video]](https://www.youtube.com/watch?v=BKgwjhao5DU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_04_text_nlp.ipynb)
# * Part 11.5: Learning English from Scratch with Keras and TensorFlow [[Video]](https://www.youtube.com/watch?v=Y1khuuSjZzc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=58) [[Notebook]](t81_558_class_11_05_english_scratch.ipynb)

# # Part 11.3: What are Embedding Layers in Keras
# 
# [Embedding Layers](https://keras.io/layers/embeddings/) are a powerful feature of Keras that allow additional information to be automatically inserted into your neural network.  In the previous section you saw that Word2Vec can expand words to a 300 dimension vector.  An embedding layer would allow you to automatically insert these 300-dimension vectors in the place of word-indexes.  
# 
# Embedding layers are often used with Natural Language Processing (NLP); however, they can be used in any instance where you wish to insert a larger vector in the place of an index value.  In some ways you can think of an embedding layer as dimension expansion. However, the hope is that these additional dimensions will provide more information to the model and provide a better score.

# ### Simple Embedding Layer Example
# 
# * **input_dim** = How large is the vocabulary?  How many categories are you encoding. This is the number of items in your "lookup table".
# * **output_dim** = How many numbers in the vector that you wish to return. 
# * **input_length** = How many items are in the input feature vector that you need to transform?
# 
# Now we create one that has a vocabulary size of 10, will reduce those values between 0-9 to 4 number vectors.  Each feature vector coming in will have 2 such features.  This neural network does nothing more than pass the embedding on to the output.  But it does let us see what the embedding is doing.

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np

model = Sequential()
embedding_layer = Embedding(input_dim=10, output_dim=4, input_length=2)
model.add(embedding_layer)
model.compile('adam', 'mse')


# Now lets query the neural network with 2 rows.

# In[2]:


input_data = np.array([
    [1,2]
])

pred = model.predict(input_data)

print(input_data.shape)
print(pred)


# In[3]:


embedding_layer.get_weights()


# ### Transferring An Embedding

# In[4]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np

embedding_lookup = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
])

model = Sequential()
embedding_layer = Embedding(input_dim=3, output_dim=3, input_length=2)
model.add(embedding_layer)
model.compile('adam', 'mse')

embedding_layer.set_weights([embedding_lookup])


# In[5]:


input_data = np.array([
    [0,1]
])

pred = model.predict(input_data)

print(input_data.shape)
print(pred)


# ### Training an Embedding

# In[6]:


from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Embedding, Dense


# In[7]:


# Define 10 resturant reviews.
reviews = [
    'Never coming back!',
    'Horrible service',
    'Rude waitress',
    'Cold food.',
    'Horrible food!',
    'Awesome',
    'Awesome service!',
    'Rocks!',
    'poor work',
    'Couldn\'t have done better']

# Define labels (1=negative, 0=positive)
labels = array([1,1,1,1,1,0,0,0,0,0])


# In[ ]:





# In[8]:


VOCAB_SIZE = 50
encoded_reviews = [one_hot(d, VOCAB_SIZE) for d in reviews]
print(f"Encoded reviews: {encoded_reviews}")


# In[9]:


MAX_LENGTH = 4

padded_reviews = pad_sequences(encoded_reviews, maxlen=MAX_LENGTH, padding='post')
print(padded_reviews)


# In[10]:


model = Sequential()
embedding_layer = Embedding(VOCAB_SIZE, 8, input_length=MAX_LENGTH)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())


# In[11]:


# fit the model
model.fit(padded_reviews, labels, epochs=100, verbose=0)


# In[12]:


print(embedding_layer.get_weights()[0].shape)
print(embedding_layer.get_weights())


# In[13]:


loss, accuracy = model.evaluate(padded_reviews, labels, verbose=0)
print(f'Accuracy: {accuracy}')


# In[ ]:




