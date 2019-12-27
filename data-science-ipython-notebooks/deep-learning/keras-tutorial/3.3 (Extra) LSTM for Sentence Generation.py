#!/usr/bin/env python
# coding: utf-8

# Credits: Forked from [deep-learning-keras-tensorflow](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio

# 
# # RNN using LSTM 
#        
# 
# 
# 

# <img src="imgs/RNN-rolled.png"/ width="80px" height="80px">

# <img src="imgs/RNN-unrolled.png"/ width="400px" height="400px">

# <img src="imgs/LSTM3-chain.png"/ width="60%">

# _source: http://colah.github.io/posts/2015-08-Understanding-LSTMs_

# In[3]:


from keras.optimizers import SGD
from keras.preprocessing.text import one_hot, text_to_word_sequence, base_filter
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing import sequence


# ### Reading blog post from data directory

# In[4]:


import os
import pickle
import numpy as np


# In[5]:


DATA_DIRECTORY = os.path.join(os.path.abspath(os.path.curdir), 'data')
print(DATA_DIRECTORY)


# In[6]:


male_posts = []
female_post = []


# In[7]:


with open(os.path.join(DATA_DIRECTORY,"male_blog_list.txt"),"rb") as male_file:
    male_posts= pickle.load(male_file)
    
with open(os.path.join(DATA_DIRECTORY,"female_blog_list.txt"),"rb") as female_file:
    female_posts = pickle.load(female_file)


# In[85]:


filtered_male_posts = list(filter(lambda p: len(p) > 0, male_posts))
filtered_female_posts = list(filter(lambda p: len(p) > 0, female_posts))


# In[86]:


# text processing - one hot builds index of the words
male_one_hot = []
female_one_hot = []
n = 30000
for post in filtered_male_posts:
    try:
        male_one_hot.append(one_hot(post, n, split=" ", filters=base_filter(), lower=True))
    except:
        continue

for post in filtered_female_posts:
    try:
        female_one_hot.append(one_hot(post,n,split=" ",filters=base_filter(),lower=True))
    except:
        continue


# In[87]:


# 0 for male, 1 for female
concatenate_array_rnn = np.concatenate((np.zeros(len(male_one_hot)),
                                        np.ones(len(female_one_hot))))


# In[88]:


from sklearn.cross_validation import train_test_split

X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(np.concatenate((female_one_hot,male_one_hot)),
                                                                    concatenate_array_rnn, 
                                                                    test_size=0.2)


# In[89]:


maxlen = 100
X_train_rnn = sequence.pad_sequences(X_train_rnn, maxlen=maxlen)
X_test_rnn = sequence.pad_sequences(X_test_rnn, maxlen=maxlen)
print('X_train_rnn shape:', X_train_rnn.shape, y_train_rnn.shape)
print('X_test_rnn shape:', X_test_rnn.shape, y_test_rnn.shape)


# In[90]:


max_features = 30000
dimension = 128
output_dimension = 128
model = Sequential()
model.add(Embedding(max_features, dimension))
model.add(LSTM(output_dimension))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[91]:


model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


# In[92]:


model.fit(X_train_rnn, y_train_rnn, batch_size=32,
          nb_epoch=4, validation_data=(X_test_rnn, y_test_rnn))


# In[93]:


score, acc = model.evaluate(X_test_rnn, y_test_rnn, batch_size=32)


# In[94]:


print(score, acc)


# # Using TFIDF Vectorizer as an input instead of one hot encoder

# In[95]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[96]:


vectorizer = TfidfVectorizer(decode_error='ignore', norm='l2', min_df=5)
tfidf_male = vectorizer.fit_transform(filtered_male_posts)
tfidf_female = vectorizer.fit_transform(filtered_female_posts)


# In[97]:


flattened_array_tfidf_male = tfidf_male.toarray()
flattened_array_tfidf_female = tfidf_male.toarray()


# In[98]:


y_rnn = np.concatenate((np.zeros(len(flattened_array_tfidf_male)),
                                        np.ones(len(flattened_array_tfidf_female))))


# In[99]:


X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(np.concatenate((flattened_array_tfidf_male, 
                                                                                    flattened_array_tfidf_female)),
                                                                    y_rnn,test_size=0.2)


# In[100]:


maxlen = 100
X_train_rnn = sequence.pad_sequences(X_train_rnn, maxlen=maxlen)
X_test_rnn = sequence.pad_sequences(X_test_rnn, maxlen=maxlen)
print('X_train_rnn shape:', X_train_rnn.shape, y_train_rnn.shape)
print('X_test_rnn shape:', X_test_rnn.shape, y_test_rnn.shape)


# In[101]:


max_features = 30000
model = Sequential()
model.add(Embedding(max_features, dimension))
model.add(LSTM(output_dimension))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[102]:


model.compile(loss='mean_squared_error',optimizer='sgd', metrics=['accuracy'])


# In[103]:


model.fit(X_train_rnn, y_train_rnn, 
          batch_size=32, nb_epoch=4,
          validation_data=(X_test_rnn, y_test_rnn))


# In[104]:


score,acc = model.evaluate(X_test_rnn, y_test_rnn, 
                           batch_size=32)


# In[105]:


print(score, acc)


# # Sentence Generation using LSTM

# In[106]:


# reading all the male text data into one string
male_post = ' '.join(filtered_male_posts)

#building character set for the male posts
character_set_male = set(male_post)
#building two indices - character index and index of character
char_indices = dict((c, i) for i, c in enumerate(character_set_male))
indices_char = dict((i, c) for i, c in enumerate(character_set_male))


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 1
sentences = []
next_chars = []
for i in range(0, len(male_post) - maxlen, step):
    sentences.append(male_post[i : i + maxlen])
    next_chars.append(male_post[i + maxlen])


# In[107]:


#Vectorisation of input
x_male = np.zeros((len(male_post), maxlen, len(character_set_male)), dtype=np.bool)
y_male = np.zeros((len(male_post), len(character_set_male)), dtype=np.bool)

print(x_male.shape, y_male.shape)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x_male[i, t, char_indices[char]] = 1
    y_male[i, char_indices[next_chars[i]]] = 1

print(x_male.shape, y_male.shape)


# In[109]:



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(character_set_male))))
model.add(Dense(len(character_set_male)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[74]:


auto_text_generating_male_model.compile(loss='mean_squared_error',optimizer='sgd')


# In[110]:


import random, sys


# In[111]:


# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i


# In[113]:


# train the model, output generated text after each iteration
for iteration in range(1,10):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_male, y_male, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(male_post) - maxlen - 1)

    for diversity in [0.2, 0.4, 0.6, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = male_post[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        for iteration in range(400):
            try:
                x = np.zeros((1, maxlen, len(character_set_male)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char
            except:
                continue
                
        print(sentence)
        print()

