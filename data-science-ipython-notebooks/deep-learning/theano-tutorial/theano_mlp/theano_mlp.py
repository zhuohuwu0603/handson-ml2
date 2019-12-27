#!/usr/bin/env python
# coding: utf-8

# # Multilayer Perceptron in Theano
# 
# Credits: Forked from [summerschool2015](https://github.com/mila-udem/summerschool2015) by mila-udem
# 
# This notebook describes how to implement the building blocks for a multilayer perceptron in Theano, in particular how to define and combine layers.
# 
# We will continue using the MNIST digits classification dataset, still using Fuel.
# 
# ## The Model
# We will focus on fully-connected layers, with an elementwise non-linearity on each hidden layer, and a softmax layer (similar to the logistic regression model) for classification on the top layer.
# 
# ### A class for hidden layers
# This class does all its work in its constructor:
# - Create and initialize shared variables for its parameters (`W` and `b`), unless there are explicitly provided. Note that the initialization scheme for `W` is the one described in [Glorot & Bengio (2010)](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).
# - Build the Theano expression for the value of the output units, given a variable for the input.
# - Store the input, output, and shared parameters as members.

# In[ ]:


import numpy
import theano
from theano import tensor

# Set lower precision float, otherwise the notebook will take too long to run
theano.config.floatX = 'float32'


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=tensor.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in Glorot & Bengio (2010)
        #        suggest that you should use 4 times larger initial weights
        #        for sigmoid compared to tanh
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = tensor.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# ### A softmax class for the output
# This class performs computations similar to what was performed in the [logistic regression tutorial](../intro_theano/logistic_regression.ipynb).
# 
# Here as well, the expression for the output is built in the class constructor, which takes the input as argument. We also add the target, `y`, and store it as an argument.

# In[ ]:


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, target, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)
        
        :type target: theano.tensor.TensorType
        :type target: column tensor that describes the target for training

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # keep track of model input and target.
        # We store a flattened (vector) version of target as y, which is easier to handle
        self.input = input
        self.target = target
        self.y = target.flatten()

        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)
    
        # class-membership probabilities
        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(input, self.W) + self.b)

        # class whose probability is maximal
        self.y_pred = tensor.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]
        

    def negative_log_likelihood(self):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        log_prob = tensor.log(self.p_y_given_x)
        log_likelihood = log_prob[tensor.arange(self.y.shape[0]), self.y]
        loss = - log_likelihood.mean()
        return loss

    def errors(self):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch
        """
        misclass_nb = tensor.neq(self.y_pred, self.y)
        misclass_rate = misclass_nb.mean()
        return misclass_rate


# ### The MLP class
# That class brings together the different parts of the model.
# 
# It also adds additional controls on the training of the full network, for instance an expression for L1 or L2 regularization (weight decay).
# 
# We can specify an arbitrary number of hidden layers, providing an empty one will reproduce the logistic regression model.

# In[ ]:


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, target, n_in, n_hidden, n_out, activation=tensor.tanh):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type target: theano.tensor.TensorType
        :type target: column tensor that describes the target for training

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: list of int
        :param n_hidden: number of hidden units in each hidden layer

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in all hidden layers
        """
        # keep track of model input and target.
        # We store a flattened (vector) version of target as y, which is easier to handle
        self.input = input
        self.target = target
        self.y = target.flatten()

        # Build all necessary hidden layers and chain them
        self.hidden_layers = []
        layer_input = input
        layer_n_in = n_in

        for nh in n_hidden:
            hidden_layer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=layer_n_in,
                n_out=nh,
                activation=activation)
            self.hidden_layers.append(hidden_layer)

            # prepare variables for next layer
            layer_input = hidden_layer.output
            layer_n_in = nh

        # The logistic regression layer gets as input the hidden units of the hidden layer,
        # and the target
        self.log_reg_layer = LogisticRegression(
            input=layer_input,
            target=target,
            n_in=layer_n_in,
            n_out=n_out)
        
        # self.params has all the parameters of the model,
        # self.weights contains only the `W` variables.
        # We also give unique name to the parameters, this will be useful to save them.
        self.params = []
        self.weights = []
        layer_idx = 0
        for hl in self.hidden_layers:
            self.params.extend(hl.params)
            self.weights.append(hl.W)
            for hlp in hl.params:
                prev_name = hlp.name
                hlp.name = 'layer' + str(layer_idx) + '.' + prev_name
            layer_idx += 1
        self.params.extend(self.log_reg_layer.params)
        self.weights.append(self.log_reg_layer.W)
        for lrp in self.log_reg_layer.params:
            prev_name = lrp.name
            lrp.name = 'layer' + str(layer_idx) + '.' + prev_name

        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = sum(abs(W).sum() for W in self.weights)

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = sum((W ** 2).sum() for W in self.weights)
    
    def negative_log_likelihood(self):
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        return self.log_reg_layer.negative_log_likelihood()

    def errors(self):
        # same holds for the function computing the number of errors
        return self.log_reg_layer.errors()


# ## Training Procedure
# We will re-use the same training algorithm: stochastic gradient descent with mini-batches, and the same early-stopping criterion. Here, the number of parameters to train is variable, and we have to wait until the MLP model is actually instantiated to have an expression for the cost and the updates.
# 
# ### Gradient and Updates
# Let us define helper functions for getting expressions for the gradient of the cost wrt the parameters, and the parameter updates. The following ones are simple, but many variations can exist, for instance:
# - regularized costs, including L1 or L2 regularization
# - more complex learning rules, such as momentum, RMSProp, ADAM, ...

# In[ ]:


def nll_grad(mlp_model):
    loss = mlp_model.negative_log_likelihood()
    params = mlp_model.params
    grads = theano.grad(loss, wrt=params)
    # Return (param, grad) pairs
    return zip(params, grads)

def sgd_updates(params_and_grads, learning_rate):
    return [(param, param - learning_rate * grad)
            for param, grad in params_and_grads]

def get_simple_training_fn(mlp_model, learning_rate):
    inputs = [mlp_model.input, mlp_model.target]
    params_and_grads = nll_grad(mlp_model)
    updates = sgd_updates(params_and_grads, learning_rate=lr)
    
    return theano.function(inputs=inputs, outputs=[], updates=updates)


# In[ ]:


def regularized_cost_grad(mlp_model, L1_reg, L2_reg):
    loss = (mlp_model.negative_log_likelihood() +
            L1_reg * mlp_model.L1 + 
            L2_reg * mlp_model.L2_sqr)
    params = mlp_model.params
    grads = theano.grad(loss, wrt=params)
    # Return (param, grad) pairs
    return zip(params, grads)

def get_regularized_training_fn(mlp_model, L1_reg, L2_reg, learning_rate):
    inputs = [mlp_model.input, mlp_model.target]
    params_and_grads = regularized_cost_grad(mlp_model, L1_reg, L2_reg)
    updates = sgd_updates(params_and_grads, learning_rate=lr)
    return theano.function(inputs, updates=updates)


# ### Testing function

# In[ ]:


def get_test_fn(mlp_model):
    return theano.function([mlp_model.input, mlp_model.target], mlp_model.errors())


# ## Training the Model
# 

# ### Training procedure
# We first need to define a few parameters for the training loop and the early stopping procedure.

# In[ ]:


import timeit
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

## early-stopping parameters tuned for 1-2 min runtime
def sgd_training(train_model, test_model, train_set, valid_set, test_set, model_name='mlp_model',
                 # maximum number of epochs
                 n_epochs=20,
                 # look at this many examples regardless
                 patience=5000,
                 # wait this much longer when a new best is found
                 patience_increase=2,
                 # a relative improvement of this much is considered significant
                 improvement_threshold=0.995,
                 batch_size=20):

    n_train_batches = train_set.num_examples // batch_size

    # Create data streams to iterate through the data.
    train_stream = Flatten(DataStream.default_stream(
        train_set, iteration_scheme=SequentialScheme(train_set.num_examples, batch_size)))
    valid_stream = Flatten(DataStream.default_stream(
        valid_set, iteration_scheme=SequentialScheme(valid_set.num_examples, batch_size)))
    test_stream = Flatten(DataStream.default_stream(
        test_set, iteration_scheme=SequentialScheme(test_set.num_examples, batch_size)))

    # go through this many minibatches before checking the network on the validation set;
    # in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        minibatch_index = 0
        for minibatch_x, minibatch_y in train_stream.get_epoch_iterator():
            train_model(minibatch_x, minibatch_y)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                for valid_xi, valid_yi in valid_stream.get_epoch_iterator():
                    validation_losses.append(test_model(valid_xi, valid_yi))
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    test_losses = []
                    for test_xi, test_yi in test_stream.get_epoch_iterator():
                        test_losses.append(test_model(test_xi, test_yi))

                    test_score = numpy.mean(test_losses)
                    print('     epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           test_score * 100.))

                    # save the best parameters
                    # build a name -> value dictionary
                    best = {param.name: param.get_value() for param in mlp_model.params}
                    numpy.savez('best_{}.npz'.format(model_name), **best)

            minibatch_index += 1
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of %f %%, '
          'with test performance %f %%' %
          (best_validation_loss * 100., test_score * 100.))

    print('The code ran for %d epochs, with %f epochs/sec (%.2fm total time)' %
          (epoch, 1. * epoch / (end_time - start_time), (end_time - start_time) / 60.))


# We then load our data set.

# In[ ]:


from fuel.datasets import MNIST

# the full set is usually (0, 50000) for train, (50000, 60000) for valid and no slice for test.
# We only selected a subset to go faster.
train_set = MNIST(which_sets=('train',), sources=('features', 'targets'), subset=slice(0, 20000))
valid_set = MNIST(which_sets=('train',), sources=('features', 'targets'), subset=slice(20000, 24000))
test_set = MNIST(which_sets=('test',), sources=('features', 'targets'))


# ### Build the Model
# Now is the time to specify and build a particular instance of the MLP. Let's start with one with a single hidden layer of 500 hidden units, and a tanh non-linearity.

# In[ ]:


rng = numpy.random.RandomState(1234)
x = tensor.matrix('x')
# The labels coming from Fuel are in a "column" format
y = tensor.icol('y')

n_in = 28 * 28
n_out = 10

mlp_model = MLP(
    rng=rng,
    input=x,
    target=y,
    n_in=n_in,
    n_hidden=[500],
    n_out=n_out,
    activation=tensor.tanh)

lr = numpy.float32(0.1)
L1_reg = numpy.float32(0)
L2_reg = numpy.float32(0.0001)

train_model = get_regularized_training_fn(mlp_model, L1_reg, L2_reg, lr)
test_model = get_test_fn(mlp_model)


# ### Launch the training phase

# In[ ]:


sgd_training(train_model, test_model, train_set, valid_set, test_set)


# ## How can we make it better?
# 
# - Max-column normalization
# - Dropout
# 

# ### ReLU activation

# In[ ]:


def relu(x):
    return x * (x > 0)

rng = numpy.random.RandomState(1234)

mlp_relu = MLP(
    rng=rng,
    input=x,
    target=y,
    n_in=n_in,
    n_hidden=[500],
    n_out=n_out,
    activation=relu)

lr = numpy.float32(0.1)
L1_reg = numpy.float32(0)
L2_reg = numpy.float32(0.0001)

train_relu = get_regularized_training_fn(mlp_relu, L1_reg, L2_reg, lr)
test_relu = get_test_fn(mlp_relu)


# In[ ]:


sgd_training(train_relu, test_relu, train_set, valid_set, test_set, model_name='mlp_relu')


# ### Momentum training (Adadelta, RMSProp, ...)

# In[ ]:


# This implements simple momentum
def get_momentum_updates(params_and_grads, lr, rho):
    res = []

    # numpy will promote (1 - rho) to float64 otherwise
    one = numpy.float32(1.)
    
    for p, g in params_and_grads:
        up = theano.shared(p.get_value() * 0)
        res.append((p, p - lr * up))
        res.append((up, rho * up + (one - rho) * g))

    return res


# This implements the parameter updates for Adadelta
def get_adadelta_updates(params_and_grads, rho):
    up2 = [theano.shared(p.get_value() * 0, name="up2 for " + p.name) for p, g in params_and_grads]
    grads2 = [theano.shared(p.get_value() * 0, name="grads2 for " + p.name) for p, g in params_and_grads]

    # This is dumb but numpy will promote (1 - rho) to float64 otherwise
    one = numpy.float32(1.)
    
    rg2up = [(rg2, rho * rg2 + (one - rho) * (g ** 2))
             for rg2, (p, g) in zip(grads2, params_and_grads)]

    updir = [-(tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6)) * g
             for (p, g), ru2, rg2 in zip(params_and_grads, up2, grads2)]

    ru2up = [(ru2, rho * ru2 + (one - rho) * (ud ** 2))
             for ru2, ud in zip(up2, updir)]

    param_up = [(p, p + ud) for (p, g), ud in zip(params_and_grads, updir)]
    
    return rg2up + ru2up + param_up

# You can try to write an RMSProp function and train the model with it.

def get_momentum_training_fn(mlp_model, L1_reg, L2_reg, lr, rho):
    inputs = [mlp_model.input, mlp_model.target]
    params_and_grads = regularized_cost_grad(mlp_model, L1_reg, L2_reg)
    updates = get_momentum_updates(params_and_grads, lr=lr, rho=rho)
    return theano.function(inputs, updates=updates)


# In[ ]:


rng = numpy.random.RandomState(1234)
x = tensor.matrix('x')
# The labels coming from Fuel are in a "column" format
y = tensor.icol('y')

n_in = 28 * 28
n_out = 10

mlp_model = MLP(
    rng=rng,
    input=x,
    target=y,
    n_in=n_in,
    n_hidden=[500],
    n_out=n_out,
    activation=tensor.tanh)

lr = numpy.float32(0.1)
L1_reg = numpy.float32(0)
L2_reg = numpy.float32(0.0001)
rho = numpy.float32(0.95)

momentum_train = get_momentum_training_fn(mlp_model, L1_reg, L2_reg, lr=lr, rho=rho)
test_fn = get_test_fn(mlp_model)

sgd_training(momentum_train, test_fn, train_set, valid_set, test_set, n_epochs=20, model_name='mlp_momentum')

