#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this demo, you'll see a more practical application of RNNs/LSTMs as character-level language models. The emphasis will be more on parallelization and using RNNs with data from Fuel.
# 
# To get started, we first need to download the training text, validation text and a file that contains a dictionary for mapping characters to integers. We also need to import quite a list of modules.

# In[6]:


import os
import requests
import gzip

from six.moves import cPickle as pkl
import time

import numpy
import theano
import theano.tensor as T

from theano.tensor.nnet import categorical_crossentropy
from theano import config
from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding

if not os.path.exists('traindata.txt'):
    r = requests.get('http://www-etud.iro.umontreal.ca/~brakelp/traindata.txt.gz')
    with open('traindata.txt.gz', 'wb') as data_file:
        data_file.write(r.content)
    with gzip.open('traindata.txt.gz', 'rb') as data_file:
        with open('traindata.txt', 'w') as out_file:
            out_file.write(data_file.read())
        
if not os.path.exists('valdata.txt'):
    r = requests.get('http://www-etud.iro.umontreal.ca/~brakelp/valdata.txt.gz')
    with open('valdata.txt.gz', 'wb') as data_file:
        data_file.write(r.content)
    with gzip.open('valdata.txt.gz', 'rb') as data_file:
        with open('valdata.txt', 'w') as out_file:
            out_file.write(data_file.read())

if not os.path.exists('dictionary.pkl'):
    r = requests.get('http://www-etud.iro.umontreal.ca/~brakelp/dictionary.pkl')
    with open('dictionary.pkl', 'wb') as data_file:
        data_file.write(r.content)


# ##The Model
# The code below shows an implementation of an LSTM network. Note that there are various different variations of the LSTM in use and this one doesn't include the so-called 'peephole connections'. We used a separate method for the dynamic update to make it easier to generate from the network later. The `index_dot` function doesn't safe much verbosity, but it clarifies that certain dot products have been replaced with indexing operations because this network will be applied to discrete data. Last but not least, note the addition of the `mask` argument which is used to ignore certain parts of the input sequence.

# In[7]:


def gauss_weight(rng, ndim_in, ndim_out=None, sd=.005):
    if ndim_out is None:
        ndim_out = ndim_in
    W = rng.randn(ndim_in, ndim_out) * sd
    return numpy.asarray(W, dtype=config.floatX)


def index_dot(indices, w):
    return w[indices.flatten()]


class LstmLayer:

    def __init__(self, rng, input, mask, n_in, n_h):

        # Init params
        self.W_i = theano.shared(gauss_weight(rng, n_in, n_h), 'W_i', borrow=True)
        self.W_f = theano.shared(gauss_weight(rng, n_in, n_h), 'W_f', borrow=True)
        self.W_c = theano.shared(gauss_weight(rng, n_in, n_h), 'W_c', borrow=True)
        self.W_o = theano.shared(gauss_weight(rng, n_in, n_h), 'W_o', borrow=True)

        self.U_i = theano.shared(gauss_weight(rng, n_h), 'U_i', borrow=True)
        self.U_f = theano.shared(gauss_weight(rng, n_h), 'U_f', borrow=True)
        self.U_c = theano.shared(gauss_weight(rng, n_h), 'U_c', borrow=True)
        self.U_o = theano.shared(gauss_weight(rng, n_h), 'U_o', borrow=True)

        self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_i', borrow=True)
        self.b_f = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_f', borrow=True)
        self.b_c = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_c', borrow=True)
        self.b_o = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_o', borrow=True)

        self.params = [self.W_i, self.W_f, self.W_c, self.W_o,
                       self.U_i, self.U_f, self.U_c, self.U_o,
                       self.b_i, self.b_f, self.b_c, self.b_o]

        outputs_info = [T.zeros((input.shape[1], n_h)),
                        T.zeros((input.shape[1], n_h))]

        rval, updates = theano.scan(self._step,
                                    sequences=[mask, input],
                                    outputs_info=outputs_info)

        # self.output is in the format (length, batchsize, n_h)
        self.output = rval[0]

    def _step(self, m_, x_, h_, c_):

        i_preact = (index_dot(x_, self.W_i) +
                    T.dot(h_, self.U_i) + self.b_i)
        i = T.nnet.sigmoid(i_preact)

        f_preact = (index_dot(x_, self.W_f) +
                    T.dot(h_, self.U_f) + self.b_f)
        f = T.nnet.sigmoid(f_preact)

        o_preact = (index_dot(x_, self.W_o) +
                    T.dot(h_, self.U_o) + self.b_o)
        o = T.nnet.sigmoid(o_preact)

        c_preact = (index_dot(x_, self.W_c) +
                    T.dot(h_, self.U_c) + self.b_c)
        c = T.tanh(c_preact)

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c


# The next block contains some code that computes cross-entropy for masked sequences and a stripped down version of the logistic regression class from the deep learning tutorials which we will need later.

# In[8]:


def sequence_categorical_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] *
                                           prediction.shape[1]),
                                          prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)


class LogisticRegression(object):
   
    def __init__(self, rng, input, n_in, n_out):
        
        W = gauss_weight(rng, n_in, n_out)
        self.W = theano.shared(value=numpy.asarray(W, dtype=theano.config.floatX),
                               name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        energy = T.dot(input, self.W) + self.b
        energy_exp = T.exp(energy - T.max(energy, axis=2, keepdims=True))
        pmf = energy_exp / energy_exp.sum(axis=2, keepdims=True)
        self.p_y_given_x = pmf
        self.params = [self.W, self.b]


# #Processing the Data
# The data in `traindata.txt` and `valdata.txt` is simply English text but formatted in such a way that every sentence is conveniently separated by the newline symbol. We'll use some of the functionality of fuel to perform the following preprocessing steps:
# * Convert everything to lowercase
# * Map characters to indices
# * Group the sentences into batches
# * Convert each batch in a matrix/tensor as long as the longest sequence with zeros padded to all the shorter sequences
# * Add a mask matrix that encodes the length of each sequence (a timestep at which the mask is 0 indicates that there is no data available)

# In[9]:


batch_size = 100
n_epochs = 40
n_h = 50
DICT_FILE = 'dictionary.pkl'
TRAIN_FILE = 'traindata.txt'
VAL_FILE = 'valdata.txt'

# Load the datasets with Fuel
dictionary = pkl.load(open(DICT_FILE, 'r'))
# add a symbol for unknown characters
dictionary['~'] = len(dictionary)
reverse_mapping = dict((j, i) for i, j in dictionary.items())

train = TextFile(files=[TRAIN_FILE],
                 dictionary=dictionary,
                 unk_token='~',
                 level='character',
                 preprocess=str.lower,
                 bos_token=None,
                 eos_token=None)

train_stream = DataStream.default_stream(train)

# organize data in batches and pad shorter sequences with zeros
train_stream = Batch(train_stream,
                     iteration_scheme=ConstantScheme(batch_size))
train_stream = Padding(train_stream)

# idem dito for the validation text
val = TextFile(files=[VAL_FILE],
                 dictionary=dictionary,
                 unk_token='~',
                 level='character',
                 preprocess=str.lower,
                 bos_token=None,
                 eos_token=None)

val_stream = DataStream.default_stream(val)

# organize data in batches and pad shorter sequences with zeros
val_stream = Batch(val_stream,
                     iteration_scheme=ConstantScheme(batch_size))
val_stream = Padding(val_stream)


# ##The Theano Graph
# We'll now define the complete Theano graph for computing costs and gradients among other things. The cost will be the cross-entropy of the next character in the sequence and the network will try to predict it based on the previous characters.

# In[10]:


# Set the random number generator' seeds for consistency
rng = numpy.random.RandomState(12345)

x = T.lmatrix('x')
mask = T.matrix('mask')

# Construct an LSTM layer and a logistic regression layer
recurrent_layer = LstmLayer(rng=rng, input=x, mask=mask, n_in=111, n_h=n_h)
logreg_layer = LogisticRegression(rng=rng, input=recurrent_layer.output[:-1],
                                  n_in=n_h, n_out=111)

# define a cost variable to optimize
cost = sequence_categorical_crossentropy(logreg_layer.p_y_given_x,
                                         x[1:],
                                         mask[1:]) / batch_size

# create a list of all model parameters to be fit by gradient descent
params = logreg_layer.params + recurrent_layer.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)


# We can now compile the function that updates the gradients. We also added a function that computes the cost without updating for monitoring purposes.

# In[11]:


learning_rate = 0.1
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]

update_model = theano.function([x, mask], cost, updates=updates)

evaluate_model = theano.function([x, mask], cost)


# ##Generating Sequences
# To see if the networks learn something useful (and to make results monitoring more entertaining), we'll also write some code to generate sequences. For this, we'll first compile a function that computes a single state update for the network to have more control over the values of each variable at each time step.

# In[12]:


x_t = T.iscalar()
h_p = T.vector()
c_p = T.vector()
h_t, c_t = recurrent_layer._step(T.ones(1), x_t, h_p, c_p)
energy = T.dot(h_t, logreg_layer.W) + logreg_layer.b

energy_exp = T.exp(energy - T.max(energy, axis=1, keepdims=True))

output = energy_exp / energy_exp.sum(axis=1, keepdims=True)
single_step = theano.function([x_t, h_p, c_p], [output, h_t, c_t])

def speak(single_step, prefix='the meaning of life is ', n_steps=450):
    try:
        h_p = numpy.zeros((n_h,), dtype=config.floatX)
        c_p = numpy.zeros((n_h,), dtype=config.floatX)
        sentence = prefix
        for char in prefix:
            x_t = dictionary[char]
            prediction, h_p, c_p = single_step(x_t, h_p.flatten(),
                                               c_p.flatten())
        # Renormalize probability in float64
        flat_prediction = prediction.flatten()
        flat_pred_sum = flat_prediction.sum(dtype='float64')
        if flat_pred_sum > 1:
            flat_prediction = flat_prediction.astype('float64') / flat_pred_sum
        sample = numpy.random.multinomial(1, flat_prediction)

        for i in range(n_steps):
            x_t = numpy.argmax(sample)
            prediction, h_p, c_p = single_step(x_t, h_p.flatten(),
                                               c_p.flatten())
            # Renormalize probability in float64
            flat_prediction = prediction.flatten()
            flat_pred_sum = flat_prediction.sum(dtype='float64')
            if flat_pred_sum > 1:
                flat_prediction = flat_prediction.astype('float64') / flat_pred_sum
            sample = numpy.random.multinomial(1, flat_prediction)

            sentence += reverse_mapping[x_t]

        return sentence
    except ValueError as e:
        print 'Something went wrong during sentence generation: {}'.format(e)


# In[13]:


start_time = time.clock()

iteration = 0

for epoch in range(n_epochs):
    print 'epoch:', epoch

    for x_, mask_ in train_stream.get_epoch_iterator():
        iteration += 1

        cross_entropy = update_model(x_.T, mask_.T)


        # Generate some text after each 20 minibatches
        if iteration % 40 == 0:
            sentence = speak(single_step, prefix='the meaning of life is ', n_steps=450)
            print
            print 'LSTM: "' + sentence + '"'
            print
            print 'epoch:', epoch, '  minibatch:', iteration
            val_scores = []
            for x_val, mask_val in val_stream.get_epoch_iterator():
                val_scores.append(evaluate_model(x_val.T, mask_val.T))
            print 'Average validation CE per sentence:', numpy.mean(val_scores)

end_time = time.clock()
print('Optimization complete.')
print('The code ran for %.2fm' % ((end_time - start_time) / 60.))


# It can take a while before the text starts to look more reasonable but here are some things to experiment with:
# * Smarter optimization algorithms (or at least momentum)
# * Initializing the recurrent weights orthogonally
# * The sizes of the initial weights and biases (think about what the gates do)
# * Different sentence prefixes
# * Changing the temperature of the character distribution during generation. What happens when you generate deterministically?

# In[ ]:




