#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 1: Python Preliminaries**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 1 Material
# 
# * **Part 1.1: Course Overview** [[Video]](https://www.youtube.com/watch?v=v8QsRio8zUM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_1_overview.ipynb)
# * Part 1.2: Introduction to Python [[Video]](https://www.youtube.com/watch?v=czq5d53vKvo&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_2_intro_python.ipynb)
# * Part 1.3: Python Lists, Dictionaries, Sets and JSON [[Video]](https://www.youtube.com/watch?v=kcGx2I5akSs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_3_python_collections.ipynb)
# * Part 1.4: File Handling [[Video]](https://www.youtube.com/watch?v=FSuSLCMgCZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_4_python_files.ipynb)
# * Part 1.5: Functions, Lambdas, and Map/Reduce [[Video]](https://www.youtube.com/watch?v=jQH1ZCSj6Ng&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_01_5_python_functional.ipynb)
# 
# Watch one (or more) of these depending on how you want to setup your Python TensorFlow environment:
# * [How to Submit a Module Assignment locally](https://www.youtube.com/watch?v=hmCGjCVhYNc)
# * [How to Use Google CoLab and Submit Assignment](https://www.youtube.com/watch?v=Pt-Od-oBgOM)
# * [Installing TensorFlow, Keras, and Python in Windows](https://www.youtube.com/watch?v=59duINoc8GM)
# * [Installing TensorFlow, Keras, and Python in Mac](https://www.youtube.com/watch?v=mcIKDJYeyFY)

# # Part 1.1: Course Overview
# 
# Deep learning is a group of exciting new technologies for neural networks. By using a combination of advanced training techniques neural network architectural components, it is now possible to train neural networks of much greater complexity. This course will introduce the student to deep belief neural networks, regularization units (ReLU), convolution neural networks and recurrent neural networks. High performance computing (HPC) aspects will demonstrate how deep learning can be leveraged both on graphical processing units (GPUs), as well as grids. Deep learning allows a model to learn hierarchies of information in a way that is similar to the function of the human brain. Focus will be primarily upon the application of deep learning, with some introduction to the mathematical foundations of deep learning. Students will use the Python programming language to architect a deep learning model for several of real-world data sets and interpret the results of these networks.

# # Assignments
# 
# Your grade will be calculated according to the following assignments:
# 
# Assignment          |Weight|Description
# --------------------|------|-------
# Class Participation |   10%|Class attendance and participation (individual)
# Class Assignments   |   50%|10 small programming assignments (5% each, individual)
# Kaggle Project      |   20%|"Kaggle In-Class" project submitted through Kaggle (Kaggle Team, up to 5 people)
# Final Project       |   20%|Deep Learning Implementation Report (Same Kaggle Team)
# 
# The 10 class assignments will be assigned with each of the first 10 modules.  Generally, each module assignment is due just before the following module date.  Refer to syllabus for exact due dates.  The 10 class assignments are submitted using the Python submission script.  Refer to assignment 1 for details.
# 
# * Module 1 Assignment: [How to Submit an Assignment](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class1.ipynb)
# * Module 2 Assignment: [Creating Columns in Pandas](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class2.ipynb)
# * Module 3 Assignment: [Data Preparation in Pandas](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class3.ipynb)
# * Module 4 Assignment: [Classification and Regression Neural Networks](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class4.ipynb)
# * Module 5 Assignment: [K-Fold Cross-Validation](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class5.ipynb)
# * Module 6 Assignment: [Image Processing](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class6.ipynb)
# * Module 7 Assignment: [Computer Vision](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class7.ipynb)
# * Module 8 Assignment: [Building a Kaggle Submission File](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class8.ipynb)
# * Module 9 Assignment: [Counting Items in a YOLO Image](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class9.ipynb)
# * Module 10 Assignment: [Time Series Neural Network](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class10.ipynb)
# 

# # Your Instructor: Jeff Heaton
# 
# ![Jeff Heaton at WUSTL Video Studio](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/jheaton1.png "Jeff Heaton")
# 
# I will be your instructor for this course.  A brief summary of my credentials is given here:
# 
# * Master of Information Management (MIM), Washington University in St. Louis, MO
# * PhD in Computer Science, Nova Southeastern University in Ft. Lauderdale, FL
# * [Vice President and Data Scientist](http://www.rgare.com/knowledge-center/media/articles/rga-where-logic-meets-curiosity), Reinsurance Group of America (RGA)
# * Senior Member, IEEE
# * jtheaton at domain name of this university
# * Other industry certifications: FLMI, ARA, ACS
# 
# Social media:
# 
# * [Homepage](http://www.heatonresearch.com) - My home page.  Includes my research interests and publications.
# * [YouTube Channel](https://www.youtube.com/user/HeatonResearch) - My YouTube Channel.  Subscribe for my videos on AI and updates to this class.
# * [GitHub](https://github.com/jeffheaton) - My GitHub repositories.
# * [Linked In](https://www.linkedin.com/in/jeffheaton) - My Linked In profile.
# * [Twitter](https://twitter.com/jeffheaton) - My Twitter feed.
# * [Google Scholar](https://scholar.google.com/citations?user=1jPGeg4AAAAJ&hl=en) - My citations on Google Scholar.
# * [Research Gate](https://www.researchgate.net/profile/Jeff_Heaton) - My profile/research at Research Gate.
# * [Others](http://www.heatonresearch.com/about/) - About me and other social media sites that I am a member of.

# # Course Resources
# 
# * [Google CoLab](https://colab.research.google.com/) - Free web-based platform that includes Python, Juypter Notebooks, and TensorFlow [[Cite:GoogleTensorFlow]](http://download.tensorflow.org/paper/whitepaper2015.pdf).  No setup needed.
# * [Python Anaconda](https://www.continuum.io/downloads) - Python distribution that includes many data science packages, such as Numpy, Scipy, Scikit-Learn, Pandas, and much more.
# * [Juypter Notebooks](http://jupyter.org/) - Easy to use environment that combines Python, Graphics and Text. 
# * [TensorFlow](https://www.tensorflow.org/) - Google's mathematics package for deep learning.
# * [Kaggle](https://www.kaggle.com/) - Competitive data science.  Good source of sample data.
# * [Course GitHub Repository](https://github.com/jeffheaton/t81_558_deep_learning) - All of the course notebooks will be published here.

# # What is Deep Learning
# 
# The focus of this class is deep learning, which is a very popular type of machine learning that is based upon the original neural networks popularized in the 1980's. There is very little difference between how a deep neural network is calculated compared with the original neural network.  We've always been able to create and calculate deep neural networks.  A deep neural network is nothing more than a neural network with many layers.  While we've always been able to create/calculate deep neural networks, we've lacked an effective means of training them.  Deep learning provides an efficient means to train deep neural networks.
# 
# ## What is Machine Learning
# 
# If deep learning is a type of machine learning, this begs the question, "What is machine learning?"  The following diagram illustrates how machine learning differs from traditional software development.
# 
# ![ML vs Traditional Software Development](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_1_ml_vs_trad.png "Machine Learning vs Traditional Software Development")
# 
# * **Traditional Software Development** - Programmers create programs that specify how to transform input into the desired output.
# * **Machine Learning** - Programmers create models that can learn to produce the desired output for given input. This learning fills the traditional role of the computer program. 
# 
# Researchers have applied machine learning to many different areas.  This class will explore three specific domains for the application of deep neural networks:
# 
# ![Application of Machine Learning](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_1_ml_types.png "Application of Machine Learning")
# 
# * **Predictive Modeling** - Several named input values are used to predict another named value that becomes the output.  For example, using four measurements of iris flowers to predict the species.  This type of data is often called tabular data.
# * **Computer Vision** - The use of machine learning to detect patterns in visual data.  For example, is an image a cat or a dog.
# * **Time Series** - The use of machine learning to detect patterns in in time.  Common applications of time series are: financial applications, speech recognition, and even natural language processing (NLP). 
# 
# ### Regression
# 
# Regression is when a model, such as a neural network, accepts input and produces a numeric output.  Consider if you were tasked to write a program that predicted how many miles per gallon (MPG) a car could achieve.  For the inputs you would probably want such features as the weight of the car, the horsepower, how large the engine is, etc.  Your program would be a combination of math and if-statements.  
# 
# Machine learning lets the computer learn the "formula" for calculating the MPG of a car, using data.  Consider [this](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/data/auto-mpg.csv) dataset.  We can use regression machine learning models to study this data and learn how to predict the MPG for a car. 
# 
# ### Classification
# 
# The output of a classification model is what class the input belongs to.  For example, consider using four measurements of an iris flower to determine the species that the flower is in.  This dataset could be used to perform [this](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/data/iris.csv).  
# 
# ### Beyond Classification and Regression
# 
# One of the most powerful aspects of neural networks is that they simply cannot be typed as either regression or classification.  The output from a neural network could be any number of the following:
# 
# * An image
# * A series of numbers that could be interpreted as text, audio, or another time series
# * A regression number
# * A classification class
# 
# ## What are Neural Networks
# 
# Neural networks are one of the earliest types of machine learning model.  Neural networks were originally introduced in the 1940's and have risen and fallen [several times from popularity](http://hushmagazine.ca/living-2/business/the-believers-the-hidden-story-behind-the-code-that-runs-our-lives). Four researchers have contributed greatly to the development of neural networks.  They have consistently pushed neural network research, both through the ups and downs: 
# 
# ![Neural Network Luminaries](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_1_luminaries_ann.png "Neural Network Luminaries")
# 
# The current luminaries of artificial neural network (ANN) research and ultimately deep learning, in order as appearing in the above picture:
# 
# * [Yann LeCun](http://yann.lecun.com/), Facebook and New York University - Optical character recognition and computer vision using convolutional neural networks (CNN).  The founding father of convolutional nets.
# * [Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/), Google and University of Toronto. Extensive work on neural networks. Creator of deep learning and early adapter/creator of backpropagation for neural networks.
# * [Yoshua Bengio](http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html), University of Montreal. Extensive research into deep learning, neural networks, and machine learning.  He has so far remained completely in academia.
# * [Andrew Ng](http://www.andrewng.org/), Badiu and Stanford University.  Extensive research into deep learning, neural networks, and application to robotics.
# 
# Geoffrey Hinton, Yann LeCun, and Yoshua Bengio won the [Turing Award](https://www.acm.org/media-center/2019/march/turing-award-2018) for their contributions to deep lerning.
# 
# ## Why Deep Learning?
# 
# For predictive modeling neural networks are not that different than other models, such as:
# 
# * Support Vector Machines
# * Random Forests
# * Gradient Boosted Machines
# 
# Like these other models, neural networks can perform both **classification** and **regression**.  When applied to relatively low-dimensional predictive modeling tasks, deep neural networks do not necessarily add significant accuracy over other model types.  Andrew Ng describes the advantage of deep neural networks over traditional model types as follows:
# 
# ![Why Deep Learning?](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_1_why_deep.png "Why Deep Learning")
# 
# Neural networks also have two additional significant advantages over other machine learning models:
# 
# * **Convolutional Neural Networks** - Can scan an image for patterns within the image.
# * **Recurrent Neural Networks** - Can find patterns across several inputs, not just within a single input.
# 
# Neural networks are also very flexible on the type of data that can be presented to the input and output layers.  A neural network can take tabular data, images, audio sequences, time series tabular data, and text as its input or output.  

# # Python for Deep Learning
# 
# Python 3.x is the programming language that will be used for this class.  Python, as a programming language, has the widest support for deep learning.  The three most popular frameworks for deep learning in Python are:
# 
# * [TensorFlow](https://www.tensorflow.org/) (Google)
# * [MXNet](https://github.com/dmlc/mxnet) (Amazon)
# * [CNTK](https://cntk.ai/) (Microsoft)
# * [Theano](http://deeplearning.net/software/theano/) (University of Montreal) - Popular but discontinued. 
# 
# Some references on popular programming languages for AI/Data Science:
# 
# * [Popular Programming Languages for AI](https://en.wikipedia.org/wiki/List_of_programming_languages_for_artificial_intelligence)
# * [Popular Programming Languages for Data Science](http://www.kdnuggets.com/2014/08/four-main-languages-analytics-data-mining-data-science.html)

# # Software Installation
# This is a technical class.  You will need to be able to compile and execute Python code that makes use of TensorFlow for deep learning. There are two options to you for accomplish this:
# 
# * Install Python, TensorFlow and some IDE (Jupyter, TensorFlow, etc.)
# * Use Google CoLab in the cloud
# 
# ## Installing Python and TensorFlow
# 
# It is possible to install and run Python/TensorFlow entirely from your own computer.  Google provides TensorFlow for Windows, Mac and Linux.  Previously, TensorFlow did not support Windows.  However, as of December 2016, TensorFlow supports Windows for both CPU and GPU operation.
# 
# The first step is to install Python 3.7.  As of August 2019, this is the latest version of Python 3.  I recommend using the Miniconda (Anaconda) release of Python, as it already includes many of the data science related packages that will be needed by this class.  Anaconda directly supports: Windows, Mac and Linux.  Miniconda is the minimal set of features from the very large Anaconda Python distribution.  Download Miniconda from the following URL:
# 
# * [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
# 
# # Dealing with TensorFlow incompatibility with Python 3.7
# 
# *Note: I will remove this section once all needed libraries add support for Python 3.7.
# 
# **VERY IMPORTANT** Once Miniconda has been downloaded you must create a Python 3.6 environment.  Not all TensorFlow 2.0 packages currently (as of August 2019) support Python 3.7.  This is not unusual, usually you will need to stay one version back from the latest Python to maximize compatibility with common machine learning packages. So you must execute the following commands:
# 
# ```
# conda create -y --name tensorflow python=3.6
# ```
# 
# To enter this environment, you must use the following command (**for Windows**), this command must be done every time you open a new Anaconda/Miniconda terminal window:
# 
# ```
# activate tensorflow
# ```
# 
# 
# For **Mac**, do this:
# 
# ```
# source activate tensorflow
# ```
# 
# # Installing Jupyter
# 
# it is easy to install Jupyter notebooks with the following command:
# 
# ```
# conda install -y jupyter
# ```
# 
# Once Jupyter is installed, it is started with the following command:
# 
# ```
# jupyter notebook
# ```
# 
# The following packages are needed for this course:
# 
# ```
# conda install -y scipy
# pip install --exists-action i --upgrade sklearn
# pip install --exists-action i --upgrade pandas
# pip install --exists-action i --upgrade pandas-datareader
# pip install --exists-action i --upgrade matplotlib
# pip install --exists-action i --upgrade pillow
# pip install --exists-action i --upgrade tqdm
# pip install --exists-action i --upgrade requests
# pip install --exists-action i --upgrade h5py
# pip install --exists-action i --upgrade pyyaml
# pip install --exists-action i --upgrade tensorflow_hub
# pip install --exists-action i --upgrade bayesian-optimization
# pip install --exists-action i --upgrade spacy
# pip install --exists-action i --upgrade gensim
# pip install --exists-action i --upgrade flask
# pip install --exists-action i --upgrade boto3
# pip install --exists-action i --upgrade gym
# pip install --exists-action i --upgrade tensorflow==2.0.0-beta1
# pip install --exists-action i --upgrade keras-rl2 --user
# conda update -y --all
# ```
# 
# Notice that I am installing as specific version of TensorFlow.  As of the current semester, this is the latest version of TensorFlow.  It is very likely that Google will upgrade this during this semester. The newer version may have some incompatibilities, so it is important that we start with this version and end with the same.
# 
# You should also link your new **tensorflow** environment to Jupyter so that you can choose it as a Kernal.  Always make sure to run your Jupyter notebooks from your 3.6 kernel.  This is demonstrated in the video.
# 
# ```
# python -m ipykernel install --user --name tensorflow --display-name "Python 3.6 (tensorflow)"
# ```
# 
# 

# # Python Introduction
# 
# 
# * [Anaconda v3.6](https://www.continuum.io/downloads) Scientific Python Distribution, including: [Scikit-Learn](http://scikit-learn.org/), [Pandas](http://pandas.pydata.org/), and others: csv, json, numpy, scipy
# * [Jupyter Notebooks](http://jupyter.readthedocs.io/en/latest/install.html)
# * [PyCharm IDE](https://www.jetbrains.com/pycharm/)
# * [Cx_Oracle](http://cx-oracle.sourceforge.net/)
# * [MatPlotLib](http://matplotlib.org/)
# 
# ## Jupyter Notebooks
# 
# Space matters in Python, indent code to define blocks
# 
# Jupyter Notebooks Allow Python and Markdown to coexist.
# 
# Even LaTeX math:
# 
# $ f'(x) = \lim_{h\to0} \frac{f(x+h) - f(x)}{h}. $
# 
# ## Python Versions
# 
# * If you see `xrange` instead of `range`, you are dealing with Python 2
# * If you see `print x` instead of `print(x)`, you are dealing with Python 2 
# * This class uses Python 3.6!

# In[1]:


# What version of Python do you have?
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


# Software used in this class:
#     
# * **Python** - The programming language.
# * **TensorFlow** - Googles deep learning framework, must have the version specified above. 
# * **Keras** - [Keras](https://github.com/fchollet/keras) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. 
# * **Pandas** - Allows for data preprocessing.  Tutorial [here](http://pandas.pydata.org/pandas-docs/version/0.18.1/tutorials.html)
# * **Scikit-Learn** - Machine learning framework for Python.  Tutorial [here](http://scikit-learn.org/stable/tutorial/basic/tutorial.html).

# # Module 1 Assignment
# 
# You can find the first assignment here: [assignment 1](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class1.ipynb)
