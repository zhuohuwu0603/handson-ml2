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
# * Part 11.3: What are Embedding Layers in Keras [[Video]](https://www.youtube.com/watch?v=OuNH5kT-aD0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_03_embedding.ipynb)
# * **Part 11.4: Natural Language Processing with Spacy and Keras** [[Video]](https://www.youtube.com/watch?v=BKgwjhao5DU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_04_text_nlp.ipynb)
# * Part 11.5: Learning English from Scratch with Keras and TensorFlow [[Video]](https://www.youtube.com/watch?v=Y1khuuSjZzc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=58) [[Notebook]](t81_558_class_11_05_english_scratch.ipynb)

# # Part 11.4: Natural Language Processing with Spacy and Keras
# 
# In this part we will see how to use Spacy and Keras together.
# 
# ### Word-Level Text Generation
# 
# There are a number of different approaches to teaching a neural network to output free-form text.  The most basic question is if you wish the neural network to learn at the word or character level.  In many ways, learning at the character level is the more interesting of the two.  The LSTM is learning construct its own words without even being shown what a word is.  We will begin with character-level text generation.  In the next module, we will see how we can use nearly the same technique to operate at the word level.  The automatic captioning that will be implemented in the next module is at the word level.
# 
# We begin by importing the needed Python packages and defining the sequence length, named **maxlen**.  Time-series neural networks always accept their input as a fixed length array.  Not all of the sequence might be used, it is common to fill extra elements with zeros.  The text will be divided into sequences of this length and the neural network will be trained to predict what comes after this sequence.

# In[1]:


from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import requests
import re


# In[2]:


import requests 

r = requests.get("https://data.heatonresearch.com/data/t81-558/text/treasure_island.txt")
raw_text = r.text.lower()

print(raw_text[0:1000])


# In[3]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(raw_text)
vocab = set()
tokenized_text = []

for token in doc:
    word = ''.join([i if ord(i) < 128 else ' ' for i in token.text])
    word = word.strip()
    if not token.is_digit         and not token.like_url         and not token.like_email:
        vocab.add(word)
        tokenized_text.append(word)
        
print(f"Vocab size: {len(vocab)}")


# The above section might have given you this error:
# 
# ```
# OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
# ```
# 
# If so, Spacy can be installed with a simple PIP install. This was included in the list of packages to install for this course.  You will need to ensure that you've installed a language with Spacy.  If you do not, you will get the following error:
# 
# To install English, use the following command:
# 
# ```
# python -m spacy download en
# ```
# 

# In[4]:


print(list(vocab)[:20])


# In[5]:


word2idx = dict((n, v) for v, n in enumerate(vocab))
idx2word = dict((n, v) for n, v in enumerate(vocab))


# In[6]:


tokenized_text = [word2idx[word] for word in tokenized_text]


# In[7]:


tokenized_text


# In[8]:


# cut the text in semi-redundant sequences of maxlen words
maxlen = 6
step = 3
sentences = []
next_words = []
for i in range(0, len(tokenized_text) - maxlen, step):
    sentences.append(tokenized_text[i: i + maxlen])
    next_words.append(tokenized_text[i + maxlen])
print('nb sequences:', len(sentences))


# In[9]:


sentences[0:5]


# In[10]:


import numpy as np

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i, t, word] = 1
    y[i, next_words[i]] = 1


# In[11]:


x.shape


# In[12]:


y.shape


# In[13]:


y[0:5]


# In[14]:


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(vocab))))
model.add(Dense(len(vocab), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[15]:


model.summary()


# In[16]:


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[17]:


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print("****************************************************************************")
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(tokenized_text) - maxlen)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('----- temperature:', temperature)

        #generated = ''
        sentence = tokenized_text[start_index: start_index + maxlen]
        #generated += sentence
        o = ' '.join([idx2word[idx] for idx in sentence])
        print(f'----- Generating with seed: "{o}"')
        #sys.stdout.write(generated)

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(vocab)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word] = 1.
                

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_word = idx2word[next_index]

            #generated += next_char
            sentence = sentence[1:]
            sentence.append(next_index)

            sys.stdout.write(next_word)
            sys.stdout.write(' ') 
            sys.stdout.flush()
        print()


# In[18]:


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])


# In[ ]:




