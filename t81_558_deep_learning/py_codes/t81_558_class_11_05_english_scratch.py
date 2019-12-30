#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 11: Natural Language Processing and Speech Recognition**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), School of Engineering and Applied Science, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 11 Material
# 
# * Part 11.1: Getting Started with Spacy in Python [[Video]](https://www.youtube.com/watch?v=A5BtU9vXzu8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_01_spacy.ipynb)
# * Part 11.2: Word2Vec and Text Classification [[Video]](https://www.youtube.com/watch?v=nWxtRlpObIs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_02_word2vec.ipynb)
# * Part 11.3: What are Embedding Layers in Keras [[Video]](https://www.youtube.com/watch?v=OuNH5kT-aD0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_03_embedding.ipynb)
# * Part 11.4: Natural Language Processing with Spacy and Keras [[Video]](https://www.youtube.com/watch?v=BKgwjhao5DU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_04_text_nlp.ipynb)
# * **Part 11.5: Learning English from Scratch with Keras and TensorFlow** [[Video]](https://www.youtube.com/watch?v=Y1khuuSjZzc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=58) [[Notebook]](t81_558_class_11_05_english_scratch.ipynb)

# # Part 11.5: Learning English from Scratch with Keras and TensorFlow
# 
# Using the above code you can create your own primitive chat bots.  A some what famous video on Youtube from Cornell University shows what happens [when two chat bots converse](https://www.youtube.com/watch?v=WnzlbyTZsQY).  Other interesting chat bot type technology:
# 
# * [CleverBot](http://www.cleverbot.com/)
# * [Computer Science Paper Generator](https://pdos.csail.mit.edu/archive/scigen/)
# 
# ### Other Resources
# * [Word Net](http://wordnet.princeton.edu/)
# * [bAbI Datasets](https://research.fb.com/downloads/babi/)
# 

# # End-To-End Memory Networks
# 
# The origional source papers for End-to-End Memory Networks:
# 
# * Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush, ["Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"](http://arxiv.org/abs/1502.05698)
# * Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, ["End-To-End Memory Networks"](http://arxiv.org/abs/1503.08895)
# 
# Other useful links for End-To-End Memory Networks
# 
# * [bAbI Datasets](https://research.fb.com/downloads/babi/)
# * [Keras End-To-End Memory Networks](https://github.com/fchollet/keras/blob/master/examples/babi_memnn.py)
# * [Online JavaScript Demo of End-to-End Memory Networks](http://yerevann.com/dmn-ui/#/)

# ## Imports and Utility Functions
# 
# The following imports are needed to create the end-to-end memory network. Neither Keras nor TensorFlow directly support End-to-End Memory Networks (yet), so it is necessary to create them using existing tools.  Several functions are needed defined here to read the bAbI dataset that we are using to train.

# In[1]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from functools import reduce
import pickle
import tarfile
import numpy as np
import re
import os
import time

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences
    that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file,
    retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (pad_sequences(inputs, maxlen=story_maxlen),
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers))


# ## Getting the Data
# 
# The data is downloaded from the Internet, if needed.
# 
# As you can see (below), this dataset contains stories and questions about those stories.  The computer is not learning these specific stories below, but rather how to read a story and answer a question about that story.  Consider the first story "Mary moved to the bathroom. John went to the hallway." the computer is not learning that Mary is in the bathroom or John is in the hallway, this changes per story.  Rather, the computer is learning to parse the story and extract information about individual people and their locations.
# 
# The computer is learning to read, at least in a limited sense.

# In[2]:


try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))


# In[3]:


# See what the data looks like

for i in range(5):
    print("Story: {}".format(' '.join(train_stories[i][0])))
    print("Query: {}".format(' '.join(train_stories[i][1])))
    print("Answer: {}".format(train_stories[i][2]))
    print("---")


# ## Building the Vocabulary
# 
# This type of neural network can only deal with a set vocabulary.  The words are indexed and each becomes a number.  Words not in the training vocabulary will not be recognized.

# In[4]:


vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')


for s in list(enumerate(vocab)):
    print(s)


# ## Building the Training and Test Data
# 
# The training data that is actually sent to the neural network is the vectorized representation of the sentences.  Each word is replaced by its vocab number.  Additionally, there are two parts to the input (x) data: story and query.  The answer (x) is a always a single vocab word number.  This is a classification network.  Any of the vocab words could potentially be the answer.  Stories can be at most 68 words and questions at most 4.  Both of these limits are automatically determined from the training data.

# In[5]:


print('Vectorizing the word sequences...')
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')


# In[6]:


# See individual training element.

print("Story (x): {}".format(inputs_train[0]))
print("Question (x): {}".format(queries_train[0]))
print("Answer: {}".format(answers_train[0]))


# ## Compile the Neural Network

# In[7]:


print('Compiling...')

# placeholders
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=64))
input_encoder_m.add(Dropout(0.3))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=query_maxlen))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Done.")


# ## Train the Neural Network
# 
# It will take some time (probably up to 1/2 hour) to train this network on a CPU.  The network is saved.  If you've previously saved the neural network, you can skip this step and load the neural network in the next step.

# In[8]:


start_time = time.time()
# train
model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          epochs=120,
          validation_data=([inputs_test, queries_test], answers_test))

# save
save_path = "./dnn/"
# save entire network to HDF5 (save everything, suggested)
model.save(os.path.join(save_path,"chatbot.h5"))
# save the vocab too, indexes must be the same
pickle.dump( vocab, open( os.path.join(save_path,"vocab.pkl"), "wb" ) )

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))


# In[9]:


# Load the model, if it exists, load vocab too
save_path = "./dnn/"
model = load_model(os.path.join(save_path,"chatbot.h5"))
vocab = pickle.load( open( os.path.join(save_path,"vocab.pkl"), "rb" ) )


# ## Evaluate Accuracy
# 
# We evaluate the accuracy, using the same technique as previous classification networks.

# In[10]:


pred = model.predict([inputs_test, queries_test])
# See what the predictions look like, they are just probabilities of each class.
print(pred)


# In[11]:


# Use argmax to turn those into actual predictions.  The class (word) with the highest
# probability is the answer.

pred = np.argmax(pred,axis=1)
print(pred)


# In[12]:


score = metrics.accuracy_score(answers_test, pred)
print("Final accuracy: {}".format(score))


# ## Adhoc Query
# 
# You might want to create your own stories and questions.  

# In[13]:


print("Remember, I only know these words: {}".format(vocab))
print()
story = "Daniel went to the hallway. Mary went to the bathroom.  Daniel went to the bedroom."
query = "Where is Sandra?"

adhoc_stories = (tokenize(story), tokenize(query), '?')

adhoc_train, adhoc_query, adhoc_answer = vectorize_stories([adhoc_stories])

pred = model.predict([adhoc_train, adhoc_query])
print(pred[0])
pred = np.argmax(pred,axis=1)
print("Answer: {}({})".format(vocab[pred[0]-1],pred))


# In[ ]:




