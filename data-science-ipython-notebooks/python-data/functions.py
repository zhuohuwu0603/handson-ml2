#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Functions

# * Functions as Objects
# * Lambda Functions
# * Closures
# * \*args, \*\*kwargs
# * Currying
# * Generators
# * Generator Expressions
# * itertools

# ## Functions as Objects

# Python treats functions as objects which can simplify data cleaning.  The following contains a transform utility class with two functions to clean strings:

# In[1]:


# get_ipython().run_cell_magic('file', 'transform_util.py', 'import re\n\n\nclass TransformUtil:\n\n    @classmethod\n    def remove_punctuation(cls, value):\n        """Removes !, #, and ?.\n        """        \n        return re.sub(\'[!#?]\', \'\', value) \n\n    @classmethod\n    def clean_strings(cls, strings, ops): \n        """General purpose method to clean strings.\n\n        Pass in a sequence of strings and the operations to perform.\n        """        \n        result = [] \n        for value in strings: \n            for function in ops: \n                value = function(value) \n            result.append(value) \n        return result')


# Below are nose tests that exercises the utility functions:

# In[2]:


# get_ipython().run_cell_magic('file', 'tests/test_transform_util.py', "from nose.tools import assert_equal\nfrom ..transform_util import TransformUtil\n\n\nclass TestTransformUtil():\n\n    states = [' Alabama ', 'Georgia!', 'Georgia', 'georgia', \\\n          'FlOrIda', 'south carolina##', 'West virginia?']\n    \n    expected_output = ['Alabama',\n                       'Georgia',\n                       'Georgia',\n                       'Georgia',\n                       'Florida',\n                       'South Carolina',\n                       'West Virginia']\n    \n    def test_remove_punctuation(self):\n        assert_equal(TransformUtil.remove_punctuation('!#?'), '')\n        \n    def test_map_remove_punctuation(self):\n        # Map applies a function to a collection\n        output = map(TransformUtil.remove_punctuation, self.states)\n        assert_equal('!#?' not in output, True)\n\n    def test_clean_strings(self):\n        clean_ops = [str.strip, TransformUtil.remove_punctuation, str.title] \n        output = TransformUtil.clean_strings(self.states, clean_ops)\n        assert_equal(output, self.expected_output)")


# Execute the nose tests in verbose mode:

# In[3]:


# get_ipython().system('nosetests tests/test_transform_util.py -v')


# ## Lambda Functions

# Lambda functions are anonymous functions and are convenient for data analysis, as data transformation functions take functions as arguments.

# Sort a sequence of strings by the number of letters:

# In[4]:


strings = ['foo', 'bar,', 'baz', 'f', 'fo', 'b', 'ba']
strings.sort(key=lambda x: len(list(x)))
strings


# ## Closures

# Closures are dynamically-genearated functions returned by another function.  The returned function has access to the variables in the local namespace where it was created.  
# 
# Closures are often used to implement decorators.  Decorators are useful to transparently wrap something with additional functionality:
# 
# ```python
# def my_decorator(fun):
#     def myfun(*params, **kwparams):
#         do_something()
#         fun(*params, **kwparams)
#     return myfun
# ```

# Each time the following closure() is called, it generates the same output:

# In[5]:


def make_closure(x):
    def closure():
        print('Secret value is: %s' % x)
    return closure

closure = make_closure(7)
closure()


# Keep track of arguments passed:

# In[6]:


def make_watcher():
    dict_seen = {}
    
    def watcher(x):
        if x in dict_seen:
            return True
        else:
            dict_seen[x] = True
            return False
        
    return watcher

watcher = make_watcher()
seq = [1, 1, 2, 3, 5, 8, 13, 2, 5, 13]
[watcher(x) for x in seq]


# ## \*args, \*\*kwargs

# \*args and \*\*kwargs are useful when you don't know how many arguments might be passed to your function or when you want to handle named arguments that you have not defined in advance.

# Print arguments and call the input function on *args:

# In[7]:


def foo(func, arg, *args, **kwargs):
    print('arg: %s', arg)
    print('args: %s', args)
    print('kwargs: %s', kwargs)
    
    print('func result: %s', func(args))

foo(sum, "foo", 1, 2, 3, 4, 5)


# ## Currying

# Currying means to derive new functions from existing ones by partial argument appilcation.  Currying is used in pandas to create specialized functions for transforming time series data.
# 
# The argument y in add_numbers is curried:

# In[8]:


def add_numbers(x, y):
    return x + y

add_seven = lambda y: add_numbers(7, y)
add_seven(3)


# The built-in functools can simplify currying with partial:

# In[9]:


from functools import partial
add_five = partial(add_numbers, 5)
add_five(2)


# ## Generators

# A generator is a simple way to construct a new iterable object.  Generators return a sequence lazily.  When you call the generator, no code is immediately executed until you request elements from the generator.
# 
# Find all the unique ways to make change for $1:

# In[10]:


def squares(n=5):
    for x in xrange(1, n + 1):
        yield x ** 2

# No code is executed
gen = squares()

# Generator returns values lazily
for x in squares():
    print x


# ## Generator Expressions
# 
# A generator expression is analogous to a comprehension.  A list comprehension is enclosed by [], a generator expression is enclosed by ():

# In[11]:


gen = (x ** 2 for x in xrange(1, 6))
for x in gen:
    print x


# ## itertools
# 
# The library itertools has a collection of generators useful for data analysis.
# 
# Function groupby takes a sequence and a key function, grouping consecutive elements in the sequence by the input function's return value (the key).  groupby returns the function's return value (the key) and a generator.

# In[12]:


import itertools
first_letter = lambda x: x[0]
strings = ['foo', 'bar', 'baz']
for letter, gen_names in itertools.groupby(strings, first_letter):
    print letter, list(gen_names)


# itertools contains many other useful functions:
# 
# | Function      | Description|
# | ------------- |-------------|
# | imap          | Generator version of map  |
# | ifilter       | Generator version of filter      |
# | combinations  | Generates a sequence of all possible k-tuples of elements in the iterable, ignoring order      |
# | permutations  | Generates a sequence of all possible k-tuples of elements in the iterable, respecting order      |
# | groupby       | Generates (key, sub-iterator) for each unique key      |
