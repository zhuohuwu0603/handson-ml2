#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 
# *No changes were made to the contents of this notebook from the original.*

# <!--NAVIGATION-->
# < [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) | [Contents](Index.ipynb) | [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb) >

# # Computation on NumPy Arrays: Universal Functions

# Up until now, we have been discussing some of the basic nuts and bolts of NumPy; in the next few sections, we will dive into the reasons that NumPy is so important in the Python data science world.
# Namely, it provides an easy and flexible interface to optimized computation with arrays of data.
# 
# Computation on NumPy arrays can be very fast, or it can be very slow.
# The key to making it fast is to use *vectorized* operations, generally implemented through NumPy's *universal functions* (ufuncs).
# This section motivates the need for NumPy's ufuncs, which can be used to make repeated calculations on array elements much more efficient.
# It then introduces many of the most common and useful arithmetic ufuncs available in the NumPy package.

# ## The Slowness of Loops
# 
# Python's default implementation (known as CPython) does some operations very slowly.
# This is in part due to the dynamic, interpreted nature of the language: the fact that types are flexible, so that sequences of operations cannot be compiled down to efficient machine code as in languages like C and Fortran.
# Recently there have been various attempts to address this weakness: well-known examples are the [PyPy](http://pypy.org/) project, a just-in-time compiled implementation of Python; the [Cython](http://cython.org) project, which converts Python code to compilable C code; and the [Numba](http://numba.pydata.org/) project, which converts snippets of Python code to fast LLVM bytecode.
# Each of these has its strengths and weaknesses, but it is safe to say that none of the three approaches has yet surpassed the reach and popularity of the standard CPython engine.
# 
# The relative sluggishness of Python generally manifests itself in situations where many small operations are being repeated – for instance looping over arrays to operate on each element.
# For example, imagine we have an array of values and we'd like to compute the reciprocal of each.
# A straightforward approach might look like this:

# In[1]:


import numpy as np
np.random.seed(0)

def compute_reciprocals(values):
    output = np.empty(len(values))
    for i in range(len(values)):
        output[i] = 1.0 / values[i]
    return output
        
values = np.random.randint(1, 10, size=5)
compute_reciprocals(values)


# This implementation probably feels fairly natural to someone from, say, a C or Java background.
# But if we measure the execution time of this code for a large input, we see that this operation is very slow, perhaps surprisingly so!
# We'll benchmark this with IPython's ``%timeit`` magic (discussed in [Profiling and Timing Code](01.07-Timing-and-Profiling.ipynb)):

# In[2]:


big_array = np.random.randint(1, 100, size=1000000)
get_ipython().run_line_magic('timeit', 'compute_reciprocals(big_array)')


# It takes several seconds to compute these million operations and to store the result!
# When even cell phones have processing speeds measured in Giga-FLOPS (i.e., billions of numerical operations per second), this seems almost absurdly slow.
# It turns out that the bottleneck here is not the operations themselves, but the type-checking and function dispatches that CPython must do at each cycle of the loop.
# Each time the reciprocal is computed, Python first examines the object's type and does a dynamic lookup of the correct function to use for that type.
# If we were working in compiled code instead, this type specification would be known before the code executes and the result could be computed much more efficiently.

# ## Introducing UFuncs
# 
# For many types of operations, NumPy provides a convenient interface into just this kind of statically typed, compiled routine. This is known as a *vectorized* operation.
# This can be accomplished by simply performing an operation on the array, which will then be applied to each element.
# This vectorized approach is designed to push the loop into the compiled layer that underlies NumPy, leading to much faster execution.
# 
# Compare the results of the following two:

# In[3]:


print(compute_reciprocals(values))
print(1.0 / values)


# Looking at the execution time for our big array, we see that it completes orders of magnitude faster than the Python loop:

# In[4]:


get_ipython().run_line_magic('timeit', '(1.0 / big_array)')


# Vectorized operations in NumPy are implemented via *ufuncs*, whose main purpose is to quickly execute repeated operations on values in NumPy arrays.
# Ufuncs are extremely flexible – before we saw an operation between a scalar and an array, but we can also operate between two arrays:

# In[5]:


np.arange(5) / np.arange(1, 6)


# And ufunc operations are not limited to one-dimensional arrays–they can also act on multi-dimensional arrays as well:

# In[6]:


x = np.arange(9).reshape((3, 3))
2 ** x


# Computations using vectorization through ufuncs are nearly always more efficient than their counterpart implemented using Python loops, especially as the arrays grow in size.
# Any time you see such a loop in a Python script, you should consider whether it can be replaced with a vectorized expression.

# ## Exploring NumPy's UFuncs
# 
# Ufuncs exist in two flavors: *unary ufuncs*, which operate on a single input, and *binary ufuncs*, which operate on two inputs.
# We'll see examples of both these types of functions here.

# ### Array arithmetic
# 
# NumPy's ufuncs feel very natural to use because they make use of Python's native arithmetic operators.
# The standard addition, subtraction, multiplication, and division can all be used:

# In[7]:


x = np.arange(4)
print("x     =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division


# There is also a unary ufunc for negation, and a ``**`` operator for exponentiation, and a ``%`` operator for modulus:

# In[8]:


print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)


# In addition, these can be strung together however you wish, and the standard order of operations is respected:

# In[9]:


-(0.5*x + 1) ** 2


# Each of these arithmetic operations are simply convenient wrappers around specific functions built into NumPy; for example, the ``+`` operator is a wrapper for the ``add`` function:

# In[10]:


np.add(x, 2)


# The following table lists the arithmetic operators implemented in NumPy:
# 
# | Operator	    | Equivalent ufunc    | Description                           |
# |---------------|---------------------|---------------------------------------|
# |``+``          |``np.add``           |Addition (e.g., ``1 + 1 = 2``)         |
# |``-``          |``np.subtract``      |Subtraction (e.g., ``3 - 2 = 1``)      |
# |``-``          |``np.negative``      |Unary negation (e.g., ``-2``)          |
# |``*``          |``np.multiply``      |Multiplication (e.g., ``2 * 3 = 6``)   |
# |``/``          |``np.divide``        |Division (e.g., ``3 / 2 = 1.5``)       |
# |``//``         |``np.floor_divide``  |Floor division (e.g., ``3 // 2 = 1``)  |
# |``**``         |``np.power``         |Exponentiation (e.g., ``2 ** 3 = 8``)  |
# |``%``          |``np.mod``           |Modulus/remainder (e.g., ``9 % 4 = 1``)|
# 
# Additionally there are Boolean/bitwise operators; we will explore these in [Comparisons, Masks, and Boolean Logic](02.06-Boolean-Arrays-and-Masks.ipynb).

# ### Absolute value
# 
# Just as NumPy understands Python's built-in arithmetic operators, it also understands Python's built-in absolute value function:

# In[11]:


x = np.array([-2, -1, 0, 1, 2])
abs(x)


# The corresponding NumPy ufunc is ``np.absolute``, which is also available under the alias ``np.abs``:

# In[12]:


np.absolute(x)


# In[13]:


np.abs(x)


# This ufunc can also handle complex data, in which the absolute value returns the magnitude:

# In[14]:


x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
np.abs(x)


# ### Trigonometric functions
# 
# NumPy provides a large number of useful ufuncs, and some of the most useful for the data scientist are the trigonometric functions.
# We'll start by defining an array of angles:

# In[15]:


theta = np.linspace(0, np.pi, 3)


# Now we can compute some trigonometric functions on these values:

# In[16]:


print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))


# The values are computed to within machine precision, which is why values that should be zero do not always hit exactly zero.
# Inverse trigonometric functions are also available:

# In[17]:


x = [-1, 0, 1]
print("x         = ", x)
print("arcsin(x) = ", np.arcsin(x))
print("arccos(x) = ", np.arccos(x))
print("arctan(x) = ", np.arctan(x))


# ### Exponents and logarithms
# 
# Another common type of operation available in a NumPy ufunc are the exponentials:

# In[18]:


x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))


# The inverse of the exponentials, the logarithms, are also available.
# The basic ``np.log`` gives the natural logarithm; if you prefer to compute the base-2 logarithm or the base-10 logarithm, these are available as well:

# In[19]:


x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))


# There are also some specialized versions that are useful for maintaining precision with very small input:

# In[20]:


x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))


# When ``x`` is very small, these functions give more precise values than if the raw ``np.log`` or ``np.exp`` were to be used.

# ### Specialized ufuncs
# 
# NumPy has many more ufuncs available, including hyperbolic trig functions, bitwise arithmetic, comparison operators, conversions from radians to degrees, rounding and remainders, and much more.
# A look through the NumPy documentation reveals a lot of interesting functionality.
# 
# Another excellent source for more specialized and obscure ufuncs is the submodule ``scipy.special``.
# If you want to compute some obscure mathematical function on your data, chances are it is implemented in ``scipy.special``.
# There are far too many functions to list them all, but the following snippet shows a couple that might come up in a statistics context:

# In[21]:


from scipy import special


# In[22]:


# Gamma functions (generalized factorials) and related functions
x = [1, 5, 10]
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))


# In[23]:


# Error function (integral of Gaussian)
# its complement, and its inverse
x = np.array([0, 0.3, 0.7, 1.0])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))


# There are many, many more ufuncs available in both NumPy and ``scipy.special``.
# Because the documentation of these packages is available online, a web search along the lines of "gamma function python" will generally find the relevant information.

# ## Advanced Ufunc Features
# 
# Many NumPy users make use of ufuncs without ever learning their full set of features.
# We'll outline a few specialized features of ufuncs here.

# ### Specifying output
# 
# For large calculations, it is sometimes useful to be able to specify the array where the result of the calculation will be stored.
# Rather than creating a temporary array, this can be used to write computation results directly to the memory location where you'd like them to be.
# For all ufuncs, this can be done using the ``out`` argument of the function:

# In[24]:


x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)


# This can even be used with array views. For example, we can write the results of a computation to every other element of a specified array:

# In[25]:


y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)


# If we had instead written ``y[::2] = 2 ** x``, this would have resulted in the creation of a temporary array to hold the results of ``2 ** x``, followed by a second operation copying those values into the ``y`` array.
# This doesn't make much of a difference for such a small computation, but for very large arrays the memory savings from careful use of the ``out`` argument can be significant.

# ### Aggregates
# 
# For binary ufuncs, there are some interesting aggregates that can be computed directly from the object.
# For example, if we'd like to *reduce* an array with a particular operation, we can use the ``reduce`` method of any ufunc.
# A reduce repeatedly applies a given operation to the elements of an array until only a single result remains.
# 
# For example, calling ``reduce`` on the ``add`` ufunc returns the sum of all elements in the array:

# In[26]:


x = np.arange(1, 6)
np.add.reduce(x)


# Similarly, calling ``reduce`` on the ``multiply`` ufunc results in the product of all array elements:

# In[27]:


np.multiply.reduce(x)


# If we'd like to store all the intermediate results of the computation, we can instead use ``accumulate``:

# In[28]:


np.add.accumulate(x)


# In[29]:


np.multiply.accumulate(x)


# Note that for these particular cases, there are dedicated NumPy functions to compute the results (``np.sum``, ``np.prod``, ``np.cumsum``, ``np.cumprod``), which we'll explore in [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb).

# ### Outer products
# 
# Finally, any ufunc can compute the output of all pairs of two different inputs using the ``outer`` method.
# This allows you, in one line, to do things like create a multiplication table:

# In[30]:


x = np.arange(1, 6)
np.multiply.outer(x, x)


# The ``ufunc.at`` and ``ufunc.reduceat`` methods, which we'll explore in [Fancy Indexing](02.07-Fancy-Indexing.ipynb), are very helpful as well.
# 
# Another extremely useful feature of ufuncs is the ability to operate between arrays of different sizes and shapes, a set of operations known as *broadcasting*.
# This subject is important enough that we will devote a whole section to it (see [Computation on Arrays: Broadcasting](02.05-Computation-on-arrays-broadcasting.ipynb)).

# ## Ufuncs: Learning More

# More information on universal functions (including the full list of available functions) can be found on the [NumPy](http://www.numpy.org) and [SciPy](http://www.scipy.org) documentation websites.
# 
# Recall that you can also access information directly from within IPython by importing the packages and using IPython's tab-completion and help (``?``) functionality, as described in [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb).

# <!--NAVIGATION-->
# < [The Basics of NumPy Arrays](02.02-The-Basics-Of-NumPy-Arrays.ipynb) | [Contents](Index.ipynb) | [Aggregations: Min, Max, and Everything In Between](02.04-Computation-on-arrays-aggregates.ipynb) >
