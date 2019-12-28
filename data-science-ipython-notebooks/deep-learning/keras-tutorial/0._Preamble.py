#!/usr/bin/env python
# coding: utf-8

# Credits: Forked from [deep-learning-keras-tensorflow](https://github.com/leriomaggio/deep-learning-keras-tensorflow) by Valerio Maggio

# <div>
#     <h1 style="text-align: center;">Deep Learning with Keras</h1>
#     <img style="text-align: left" src="imgs/keras-logo-small.jpg" width="10%" />
# <div>
# 
# <div>
#     <h2 style="text-align: center;">Tutorial @ EuroScipy 2016</h2>
#     <img style="text-align: left" src="imgs/euroscipy_2016_logo.png" width="40%" />
# </div>    

# ##### Yam Peleg,  Valerio Maggio

# # Goal of this Tutorial
# 
# - **Introduce** main features of Keras
# - **Learn** how simple and Pythonic is doing Deep Learning with Keras
# - **Understand** how easy is to do basic and *advanced* DL models in Keras;
#     - **Examples and Hand-on Excerises** along the way.

# ## Source
# 
# https://github.com/leriomaggio/deep-learning-keras-euroscipy2016/

# ---

# # (Tentative) Schedule 
# 
# ## Attention: Spoilers Warning!
# 
# 
# - **Setup** (`10 mins`)
# 
# - **Part I**: **Introduction** (`~65 mins`)
# 
#     - Intro to ANN (`~20 mins`)
#         - naive pure-Python implementation
#         - fast forward, sgd, backprop
#         
#     - Intro to Theano (`15 mins`)
#         - Model + SGD with Theano
#         
#     - Introduction to Keras (`30 mins`)
#         - Overview and main features
#             - Theano backend
#             - Tensorflow backend
#         - Multi-Layer Perceptron and Fully Connected
#             - Examples with `keras.models.Sequential` and `Dense`
#             - HandsOn: MLP with keras
#             
# - **Coffe Break** (`30 mins`)
# 
# - **Part II**: **Supervised Learning and Convolutional Neural Nets** (`~45 mins`)
#     
#     - Intro: Focus on Image Classification (`5 mins`)
# 
#     - Intro to CNN (`25 mins`)
#         - meaning of convolutional filters
#             - examples from ImageNet    
#         - Meaning of dimensions of Conv filters (through an exmple of ConvNet) 
#         - Visualising ConvNets
#         - HandsOn: ConvNet with keras 
# 
#     - Advanced CNN (`10 mins`)
#         - Dropout
#         - MaxPooling
#         - Batch Normalisation
#         
#     - Famous Models in Keras (likely moved somewhere else) (`10 mins`)
#         (ref: https://github.com/fchollet/deep-learning-models)
#             - VGG16
#             - VGG19
#             - ResNet50
#             - Inception v3
#         - HandsOn: Fine tuning a network on new dataset 
#         
# - **Part III**: **Unsupervised Learning** (`10 mins`)
# 
#     - AutoEncoders (`5 mins`)
#     - word2vec & doc2vec (gensim) & `keras.datasets` (`5 mins`)
#         - `Embedding`
#         - word2vec and CNN
#     - Exercises
# 
# - **Part IV**: **Advanced Materials** (`20 mins`)
#     - RNN and LSTM (`10 mins`)
#         -  RNN, LSTM, GRU  
#     - Example of RNN and LSTM with Text (`~10 mins`) -- *Tentative*
#     - HandsOn: IMDB
# 
# - **Wrap up and Conclusions** (`5 mins`)

# ---

# # Requirements

# This tutorial requires the following packages:
# 
# - Python version 3.4+ 
#     - likely Python 2.7 would be fine, but *who knows*? :P
# - `numpy` version 1.10 or later: http://www.numpy.org/
# - `scipy` version 0.16 or later: http://www.scipy.org/
# - `matplotlib` version 1.4 or later: http://matplotlib.org/
# - `pandas` version 0.16 or later: http://pandas.pydata.org
# - `scikit-learn` version 0.15 or later: http://scikit-learn.org
# - `keras` version 1.0 or later: http://keras.io
# - `theano` version 0.8 or later: http://deeplearning.net/software/theano/
# - `ipython`/`jupyter` version 4.0 or later, with notebook support
# 
# (Optional but recommended):
# 
# - `pyyaml`
# - `hdf5` and `h5py` (required if you use model saving/loading functions in keras)
# - **NVIDIA cuDNN** if you have NVIDIA GPUs on your machines.
#     [https://developer.nvidia.com/rdp/cudnn-download]()
# 
# The easiest way to get (most) these is to use an all-in-one installer such as [Anaconda](http://www.continuum.io/downloads) from Continuum. These are available for multiple architectures.

# ---

# ### Python Version

# I'm currently running this tutorial with **Python 3** on **Anaconda**

# In[1]:


# get_ipython().system('python --version')


# # How to set up your environment

# The quickest and simplest way to setup the environment is to use [conda](https://store.continuum.io) environment manager. 
# 
# We provide in the materials a `deep-learning.yml` that is complete and **ready to use** to set up your virtual environment with conda.

# In[3]:


# get_ipython().system('cat deep-learning.yml')


# # Recreate the Conda Environment

# #### A. Create the Environment
# 
# ```
# conda env create -f deep-learning.yml  # this file is for Linux channels.
# ```
# 
# If you're using a **Mac OSX**, we also provided in the repo the conda file 
# that is compatible with `osx-channels`:
# 
# ```
# conda env create -f deep-learning-osx.yml  # this file is for OSX channels.
# ```
# 
# #### B. Activate the new `deep-learning` Environment
# 
# ```
# source activate deep-learning
# ```

# ## Optionals

# ### 1. Enabling Conda-Forge

# It is strongly suggested to enable [**conda forge**](https://conda-forge.github.io/) in your Anaconda installation.
# 
# **Conda-Forge** is a github organisation containing repositories of conda recipies.
# 
# To add `conda-forge` as an additional anaconda channel it is just required to type:
# 
# ```shell
# conda config --add channels conda-forge
# ```

# ### 2. Configure Theano
# 
# 1) Create the `theanorc` file:
# 
# ```shell
# touch $HOME/.theanorc
# ```
# 
# 2) Copy the following content into the file:
# 
# ```
# [global]
# floatX = float32
# device = gpu  # switch to cpu if no GPU is available on your machine
# 
# [nvcc]
# fastmath = True
# 
# [lib]
# cnmem=.90
# ```

# **More on [theano documentation](http://theano.readthedocs.io/en/latest/library/config.html)**

# ### 3. Installing Tensorflow as backend 

# ```shell
# # Ubuntu/Linux 64-bit, GPU enabled, Python 3.5
# # Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp35-cp35m-linux_x86_64.whl
# 
# pip install --ignore-installed --upgrade $TF_BINARY_URL
# ```

# **More on [tensorflow documentation](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html)**

# ---

# # Test if everything is up&running

# ## 1. Check import

# In[2]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


# In[3]:


import keras


# ## 2. Check installeded Versions

# In[4]:


import numpy
print('numpy:', numpy.__version__)

import scipy
print('scipy:', scipy.__version__)

import matplotlib
print('matplotlib:', matplotlib.__version__)

import IPython
print('iPython:', IPython.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)


# In[6]:


import keras
print('keras: ', keras.__version__)

import theano
print('Theano: ', theano.__version__)

# optional
import tensorflow as tf
print('Tensorflow: ', tf.__version__)


# <br>
# <h1 style="text-align: center;">If everything worked till down here, you're ready to start!</h1>

# ---
# 

# # Consulting Material

# You have two options to go through the material presented in this tutorial:
# 
# * Read (and execute) the material as **iPython/Jupyter** notebooks
# * (just) read the material as (HTML) slides

# In the first case, all you need to do is just execute `ipython notebook` (or `jupyter notebook`) depending on the version of `iPython` you have installed on your machine
# 
# (`jupyter` command works in case you have `iPython 4.0.x` installed)

# In the second case, you may simply convert the provided notebooks in `HTML` slides and see them into your browser
# thanks to `nbconvert`.
# 
# Thus, move to the folder where notebooks are stored and execute the following command:
# 
#     jupyter nbconvert --to slides ./*.ipynb --post serve
#     
#    
# (Please substitute `jupyter` with `ipython` in the previous command if you have `iPython 3.x` installed on your machine)

# ## In case...

# ..you wanna do **both** (interactive and executable slides), I highly suggest to install the terrific `RISE` ipython notebook extension: [https://github.com/damianavila/RISE](https://github.com/damianavila/RISE)
