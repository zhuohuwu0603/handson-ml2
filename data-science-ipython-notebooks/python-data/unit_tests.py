#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Nose Unit Tests with IPython Notebook

# ## Nose
# 
# Testing is a vital part of software development.  Nose extends unittest to make testing easier.

# ## Install Nose
# 
# Run the following command line:

# In[ ]:


# get_ipython().system('pip install nose')


# ## Create the Code
# 
# Save your code to a file with the %%file magic:

# In[1]:


# get_ipython().run_cell_magic('file', 'type_util.py', 'class TypeUtil:\n\n    @classmethod\n    def is_iterable(cls, obj):\n        """Determines if obj is iterable.\n\n        Useful when writing functions that can accept multiple types of\n        input (list, tuple, ndarray, iterator).  Pairs well with\n        convert_to_list.\n        """\n        try:\n            iter(obj)\n            return True\n        except TypeError:\n            return False\n\n    @classmethod\n    def convert_to_list(cls, obj):\n        """Converts obj to a list if it is not a list and it is iterable, \n        else returns the original obj.\n        """\n        if not isinstance(obj, list) and cls.is_iterable(obj):\n            obj = list(obj)\n        return obj')


# ## Create the Nose Tests
# 
# Save your test to a file with the %%file magic:

# In[2]:


# get_ipython().run_cell_magic('file', 'tests/test_type_util.py', "from nose.tools import assert_equal\nfrom ..type_util import TypeUtil\n\n\nclass TestUtil():\n\n    def test_is_iterable(self):\n        assert_equal(TypeUtil.is_iterable('foo'), True)\n        assert_equal(TypeUtil.is_iterable(7), False)\n\n    def test_convert_to_list(self):\n        assert_equal(isinstance(TypeUtil.convert_to_list('foo'), list), True)\n        assert_equal(isinstance(TypeUtil.convert_to_list(7), list), False)")


# ## Run the Nose Tests
# 
# Run the following command line:

# In[3]:


# get_ipython().system('nosetests tests/test_type_util.py -v')

