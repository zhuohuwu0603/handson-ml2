#!/usr/bin/env python
# coding: utf-8

# # Introduction to Scan in Theano
# 
# Credits: Forked from [summerschool2015](https://github.com/mila-udem/summerschool2015) by mila-udem
# 
# ## In short
# 
# * Mechanism to perform loops in a Theano graph
# * Supports nested loops and reusing results from previous iterations 
# * Highly generic
# 
# ## Implementation
# 
# A Theano function graph is composed of two types of nodes; Variable nodes which represent data and Apply node which apply Ops (which represent some computation) to Variables to produce new Variables.
# 
# From this point of view, a node that applies a Scan op is just like any other. Internally, however, it is very different from most Ops.
# 
# Inside a Scan op is yet another Theano graph which represents the computation to be performed at every iteration of the loop. During compilation, that graph is compiled into a function and, during execution, the Scan op will call that function repeatedly on its inputs to produce its outputs.
# 
# ## Example 1 : As simple as it gets
# 
# Scan's interface is complex and, thus, best introduced by examples. So, let's dive right in and start with a simple example; perform an element-wise multiplication between two vectors. 
# 
# This particular example is simple enough that Scan is not the best way to do things but we'll gradually work our way to more complex examples where Scan gets more interesting.
# 
# Let's first setup our use case by defining Theano variables for the inputs :

# In[ ]:


import theano
import theano.tensor as T
import numpy as np

vector1 = T.vector('vector1')
vector2 = T.vector('vector2')


# Next, we call the `scan()` function. It has many parameters but, because our use case is simple, we only need two of them. We'll introduce other parameters in the next examples.
# 
# The parameter `sequences` allows us to specify variables that Scan should iterate over as it loops. The first iteration will take as input the first element of every sequence, the second iteration will take as input the second element of every sequence, etc. These individual element have will have one less dimension than the original sequences. For example, for a matrix sequence, the individual elements will be vectors.
# 
# The parameter `fn` receives a function or lambda expression that expresses the computation to do at every iteration. It operates on the symbolic inputs to produce symbolic outputs. It will **only ever be called once**, to assemble the Theano graph used by Scan at every the iterations.
# 
# Since we wish to iterate over both `vector1` and `vector2` simultaneously, we provide them as sequences. This means that every iteration will operate on two inputs: an element from `vector1` and the corresponding element from `vector2`. 
# 
# Because what we want is the elementwise product between the vectors, we provide a lambda expression that, given an element `a` from `vector1` and an element `b` from `vector2` computes and return the product.

# In[ ]:


output, updates = theano.scan(fn=lambda a, b : a * b,
                              sequences=[vector1, vector2])


# Calling `scan()`, we see that it returns two outputs.
# 
# The first output contains the outputs of `fn` from every timestep concatenated into a tensor. In our case, the output of a single timestep is a scalar so output is a vector where `output[i]` is the output of the i-th iteration.
# 
# The second output details if and how the execution of the Scan updates any shared variable in the graph. It should be provided as an argument when compiling the Theano function.

# In[ ]:


f = theano.function(inputs=[vector1, vector2],
                    outputs=output,
                    updates=updates)


# If `updates` is omitted, the state of any shared variables modified by Scan will not be updated properly. Random number sampling, for instance, relies on shared variables. If `updates` is not provided, the state of the random number generator won't be updated properly and the same numbers might be sampled repeatedly. **Always** provide `updates` when compiling your Theano function.
# 
# Now that we've defined how to do elementwise multiplication with Scan, we can see that the result is as expected :

# In[ ]:


vector1_value = np.arange(0, 5).astype(theano.config.floatX) # [0,1,2,3,4]
vector2_value = np.arange(1, 6).astype(theano.config.floatX) # [1,2,3,4,5]
print(f(vector1_value, vector2_value))


# An interesting thing is that we never explicitly told Scan how many iteration it needed to run. It was automatically inferred; when given sequences, Scan will run as many iterations as the length of the shortest sequence : 

# In[ ]:


print(f(vector1_value, vector2_value[:4]))


# ## Example 2 : Non-sequences
# 
# In this example, we introduce another of Scan's features; non-sequences. To demonstrate how to use them, we use Scan to compute the activations of  a linear MLP layer over a minibatch.
# 
# It is not yet a use case where Scan is truly useful but it introduces a requirement that sequences cannot fulfill; if we want to use Scan to iterate over the minibatch elements and compute the activations for each of them, then we need some variables (the parameters of the layer), to be available 'as is' at every iteration of the loop. We do *not* want Scan to iterate over them and give only part of them at every iteration.
# 
# Once again, we begin by setting up our Theano variables :

# In[ ]:


X = T.matrix('X') # Minibatch of data
W = T.matrix('W') # Weights of the layer
b = T.vector('b') # Biases of the layer


# For the sake of variety, in this example we define the computation to be done at every iteration of the loop using a Python function, `step()`, instead of a lambda expression.
# 
# To have the full weight matrix W and the full bias vector b available at every iteration, we use the argument non_sequences. Contrary to sequences, non-sequences are not iterated upon by Scan. Every non-sequence is passed as input to every iteration.
# 
# This means that our `step()` function will need to operate on three symbolic inputs; one for our sequence X and one for each of our non-sequences W and b. 
# 
# The inputs that correspond to the non-sequences are **always** last and in the same order at the non-sequences are provided to Scan. This means that the correspondence between the inputs of the `step()` function and the arguments to `scan()` is the following : 
# 
# * `v` : individual element of the sequence `X` 
# * `W` and `b` : non-sequences `W` and `b`, respectively

# In[ ]:


def step(v, W, b):
    return T.dot(v, W) + b

output, updates = theano.scan(fn=step,
                              sequences=[X],
                              non_sequences=[W, b])


# We can now compile our Theano function and see that it gives the expected results.

# In[ ]:


f = theano.function(inputs=[X, W, b],
                    outputs=output,
                    updates=updates)

X_value = np.arange(-3, 3).reshape(3, 2).astype(theano.config.floatX)
W_value = np.eye(2).astype(theano.config.floatX)
b_value = np.arange(2).astype(theano.config.floatX)
print(f(X_value, W_value, b_value))


# ## Example 3 : Reusing outputs from the previous iterations
# 
# In this example, we will use Scan to compute a cumulative sum over the first dimension of a matrix $M$. This means that the output will be a matrix $S$ in which the first row will be equal to the first row of $M$, the second row will be equal to the sum of the two first rows of $M$, and so on.
# 
# Another way to express this, which is the way we will implement here, is that $S[t] = S[t-1] + M[t]$. Implementing this with Scan would involve iterating over the rows of the matrix $M$ and, at every iteration, reuse the cumulative row that was output at the previous iteration and return the sum of it and the current row of $M$.
# 
# If we assume for a moment that we can get Scan to provide the output value from the previous iteration as an input for every iteration, implementing a step function is simple :

# In[ ]:


def step(m_row, cumulative_sum):
    return m_row + cumulative_sum


# The trick part is informing Scan that our step function expects as input the output of a previous iteration. To achieve this, we need to use a new parameter of the `scan()` function: `outputs_info`. This parameter is used to tell Scan how we intend to use each of the outputs that are computed at each iteration.
# 
# This parameter can be omitted (like we did so far) when the step function doesn't depend on any output of a previous iteration. However, now that we wish to have recurrent outputs, we need to start using it.
# 
# `outputs_info` takes a sequence with one element for every output of the `step()` function :
# * For a **non-recurrent output** (like in every example before this one), the element should be `None`.
# * For a **simple recurrent output** (iteration $t$ depends on the value at iteration $t-1$), the element must be a tensor. Scan will interpret it as being an initial state for a recurrent output and give it as input to the first iteration, pretending it is the output value from a previous iteration. For subsequent iterations, Scan will automatically handle giving the previous output value as an input.
# 
# The `step()` function needs to expect one additional input for each simple recurrent output. These inputs correspond to outputs from previous iteration and are **always** after the inputs that correspond to sequences but before those that correspond to non-sequences. The are received by the `step()` function in the order in which the recurrent outputs are declared in the outputs_info sequence.

# In[ ]:


M = T.matrix('X')
s = T.vector('s') # Initial value for the cumulative sum

output, updates = theano.scan(fn=step,
                              sequences=[M],
                              outputs_info=[s])


# We can now compile and test the Theano function :

# In[ ]:


f = theano.function(inputs=[M, s],
                    outputs=output,
                    updates=updates)

M_value = np.arange(9).reshape(3, 3).astype(theano.config.floatX)
s_value = np.zeros((3, ), dtype=theano.config.floatX)
print(f(M_value, s_value))


# An important thing to notice here, is that the output computed by the Scan does **not** include the initial state that we provided. It only outputs the states that it has computed itself.
# 
# If we want to have both the initial state and the computed states in the same Theano variable, we have to join them ourselves.

# ## Example 4 : Reusing outputs from multiple past iterations
# 
# The Fibonacci sequence is a sequence of numbers F where the two first numbers both 1 and every subsequence number is defined as such : $F_n = F_{n-1} + F_{n-2}$. Thus, the Fibonacci sequence goes : 1, 1, 2, 3, 5, 8, 13, ...
# 
# In this example, we will cover how to compute part of the Fibonacci sequence using Scan. Most of the tools required to achieve this have been introduced in the previous examples. The only one missing is the ability to use, at iteration $i$, outputs from iterations older than $i-1$.
# 
# Also, since every example so far had only one output at every iteration of the loop, we will also compute, at each timestep, the ratio between the new term of the Fibonacci sequence and the previous term.
# 
# Writing an appropriate step function given two inputs, representing the two previous terms of the Fibonacci sequence, is easy:

# In[ ]:


def step(f_minus2, f_minus1):
    new_f = f_minus2 + f_minus1
    ratio = new_f / f_minus1
    return new_f, ratio


# The next step is defining the value of `outputs_info`.
# 
# Recall that, for **non-recurrent outputs**, the value is `None` and, for **simple recurrent outputs**, the value is a single initial state. For **general recurrent outputs**, where iteration $t$ may depend on multiple past values, the value is a dictionary. That dictionary has two values:
# * taps : list declaring which previous values of that output every iteration will need. `[-3, -2, -1]` would mean every iteration should take as input the last 3 values of that output. `[-2]` would mean every iteration should take as input the value of that output from two iterations ago.
# * initial : tensor of initial values. If every initial value has $n$ dimensions, `initial` will be a single tensor of $n+1$ dimensions with as many initial values as the oldest requested tap. In the case of the Fibonacci sequence, the individual initial values are scalars so the `initial` will be a vector. 
# 
# In our example, we have two outputs. The first output is the next computed term of the Fibonacci sequence so every iteration should take as input the two last values of that output. The second output is the ratio between successive terms and we don't reuse its value so this output is non-recurrent. We define the value of `outputs_info` as such :

# In[ ]:


f_init = T.fvector()
outputs_info = [dict(initial=f_init, taps=[-2, -1]),
                None]


# Now that we've defined the step function and the properties of our outputs, we can call the `scan()` function. Because the `step()` function has multiple outputs, the first output of `scan()` function will be a list of tensors: the first tensor containing all the states of the first output and the second tensor containing all the states of the second input.
# 
# In every previous example, we used sequences and Scan automatically inferred the number of iterations it needed to run from the length of these
# sequences. Now that we have no sequence, we need to explicitly tell Scan how many iterations to run using the `n_step` parameter. The value can be real or symbolic.

# In[ ]:


output, updates = theano.scan(fn=step,
                              outputs_info=outputs_info,
                              n_steps=10)

next_fibonacci_terms = output[0]
ratios_between_terms = output[1]


# Let's compile our Theano function which will take a vector of consecutive values from the Fibonacci sequence and compute the next 10 values :

# In[ ]:


f = theano.function(inputs=[f_init],
                    outputs=[next_fibonacci_terms, ratios_between_terms],
                    updates=updates)

out = f([1, 1])
print(out[0])
print(out[1])


# ## Precisions about the order of the arguments to the step function
# 
# When we start using many sequences, recurrent outputs and non-sequences, it's easy to get confused regarding the order in which the step function receives the corresponding inputs. Below is the full order:
# 
# * Element from the first sequence
# * ...
# * Element from the last sequence
# * First requested tap from first recurrent output
# * ...
# * Last requested tap from first recurrent output
# * ...
# * First requested tap from last recurrent output
# * ...
# * Last requested tap from last recurrent output
# * First non-sequence
# * ...
# * Last non-sequence

# ## When to use Scan and when not to
# 
# Scan is not appropriate for every problem. Here's some information to help you figure out if Scan is the best solution for a given use case.
# 
# ### Execution speed
# 
# Using Scan in a Theano function typically makes it slighly slower compared to the equivalent Theano graph in which the loop is unrolled. Both of these approaches tend to be much slower than a vectorized implementation in which large chunks of the computation can be done in parallel.
# 
# ### Compilation speed
# 
# Scan also adds an overhead to the compilation, potentially making it slower, but using it can also dramatically reduce the size of your graph, making compilation much faster. In the end, the effect of Scan on compilation speed will heavily depend on the size of the graph with and without Scan.
# 
# The compilation speed of a Theano function using Scan will usually be comparable to one in which the loop is unrolled if the number of iterations is small. It the number of iterations is large, however, the compilation will usually be much faster with Scan.
# 
# ### In summary
# 
# If you have one of the following cases, Scan can help :
# * A vectorized implementation is not possible (due to the nature of the computation and/or memory usage)
# * You want to do a large or variable number of iterations
# 
# If you have one of the following cases, you should consider other options :
# * A vectorized implementation could perform the same computation => Use the vectorized approach. It will often be faster during both compilation and execution.
# * You want to do a small, fixed, number of iterations (ex: 2 or 3) => It's probably better to simply unroll the computation

# ## Exercises
# 
# ### Exercise 1 - Computing a polynomial
# 
# In this exercise, the initial version already works. It computes the value of a polynomial ($n_0 + n_1 x + n_2 x^2 + ... $) of at most 10000 degrees given the coefficients of the various terms and the value of x.
# 
# You must modify it such that the reduction (the sum() call) is done by Scan.

# In[ ]:


coefficients = theano.tensor.vector("coefficients")
x = T.scalar("x")
max_coefficients_supported = 10000

def step(coeff, power, free_var):
    return coeff * free_var ** power

# Generate the components of the polynomial
full_range=theano.tensor.arange(max_coefficients_supported)
components, updates = theano.scan(fn=step,
                                  outputs_info=None,
                                  sequences=[coefficients, full_range],
                                  non_sequences=x)

polynomial = components.sum()
calculate_polynomial = theano.function(inputs=[coefficients, x],
                                       outputs=polynomial,
                                       updates=updates)

test_coeff = np.asarray([1, 0, 2], dtype=theano.config.floatX)
print(calculate_polynomial(test_coeff, 3))
# 19.0


# **Solution** : run the cell below to display the solution to this exercise.

# In[ ]:


get_ipython().run_line_magic('load', 'scan_ex1_solution.py')


# ### Exercise 2 - Sampling without replacement
# 
# In this exercise, the goal is to implement a Theano function that :
# * takes as input a vector of probabilities and a scalar
# * performs sampling without replacements from those probabilities as many times as the value of the scalar
# * returns a vector containing the indices of the sampled elements.
# 
# Partial code is provided to help with the sampling of random numbers since this is not something that was covered in this tutorial.

# In[ ]:


probabilities = T.vector()
nb_samples = T.iscalar()

rng = T.shared_randomstreams.RandomStreams(1234)

def sample_from_pvect(pvect):
    """ Provided utility function: given a symbolic vector of
    probabilities (which MUST sum to 1), sample one element
    and return its index.
    """
    onehot_sample = rng.multinomial(n=1, pvals=pvect)
    sample = onehot_sample.argmax()
    return sample

def set_p_to_zero(pvect, i):
    """ Provided utility function: given a symbolic vector of
    probabilities and an index 'i', set the probability of the
    i-th element to 0 and renormalize the probabilities so they
    sum to 1.
    """
    new_pvect = T.set_subtensor(pvect[i], 0.)
    new_pvect = new_pvect / new_pvect.sum()
    return new_pvect
    

# TODO use Scan to sample from the vector of probabilities and
# symbolically obtain 'samples' the vector of sampled indices.
samples = None

# Compiling the function
f = theano.function(inputs=[probabilities, nb_samples],
                    outputs=[samples])

# Testing the function
test_probs = np.asarray([0.6, 0.3, 0.1], dtype=theano.config.floatX)
for i in range(10):
    print(f(test_probs, 2))


# **Solution** : run the cell below to display the solution to this exercise.

# In[ ]:


get_ipython().run_line_magic('load', 'scan_ex2_solution.py')


# In[ ]:




