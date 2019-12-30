#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 8: Kaggle Data Sets**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 8 Material
# 
# * **Part 8.1: Introduction to Kaggle** [[Video]](https://www.youtube.com/watch?v=v4lJBhdCuCU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_1_kaggle_intro.ipynb)
# * Part 8.2: Building Ensembles with Scikit-Learn and Keras [[Video]](https://www.youtube.com/watch?v=LQ-9ZRBLasw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_2_keras_ensembles.ipynb)
# * Part 8.3: How Should you Architect Your Keras Neural Network: Hyperparameters [[Video]](https://www.youtube.com/watch?v=1q9klwSoUQw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_3_keras_hyperparameters.ipynb)
# * Part 8.4: Bayesian Hyperparameter Optimization for Keras [[Video]](https://www.youtube.com/watch?v=sXdxyUCCm8s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_08_4_bayesian_hyperparameter_opt.ipynb)
# * Part 8.5: Current Semester's Kaggle [[Video]](https://www.youtube.com/watch?v=48OrNYYey5E) [[Notebook]](t81_558_class_08_5_kaggle_project.ipynb)
# 

# # Part 8.1: Introduction to Kaggle
# 
# [Kaggle](http://www.kaggle.com) runs competitions in which data scientists compete in order to provide the best model to fit the data. A common project to get started with Kaggle is the [Titanic data set](https://www.kaggle.com/c/titanic-gettingStarted). Most Kaggle competitions end on a specific date. Website organizers have currently scheduled the Titanic competition to end on December 31, 20xx (with the year usually rolling forward). However, they have already extended the deadline several times, and an extension beyond 2014 is also possible. Second, the Titanic data set is considered a tutorial data set. In other words, there is no prize, and your score in the competition does not count towards becoming a Kaggle Master. 

# ### Kaggle Ranks
# 
# Kaggle ranks are achieved by earning gold, silver and bronze medals.
# 
# * [Kaggle Top Users](https://www.kaggle.com/rankings)
# * [Current Top Kaggle User's Profile Page](https://www.kaggle.com/stasg7)
# * [Jeff Heaton's (your instructor) Kaggle Profile](https://www.kaggle.com/jeffheaton)
# * [Current Kaggle Ranking System](https://www.kaggle.com/progression)

# ### Typical Kaggle Competition
# 
# A typical Kaggle competition will have several components.  Consider the Titanic tutorial:
# 
# * [Competition Summary Page](https://www.kaggle.com/c/titanic)
# * [Data Page](https://www.kaggle.com/c/titanic/data)
# * [Evaluation Description Page](https://www.kaggle.com/c/titanic/details/evaluation)
# * [Leaderboard](https://www.kaggle.com/c/titanic/leaderboard)
# 
# ### How Kaggle Competitions are Scored
# 
# Kaggle is provided with a data set by the competition sponsor.  This data set is divided up as follows:
# 
# * **Complete Data Set** - This is the complete data set.
#     * **Training Data Set** - You are provided both the inputs and the outcomes for the training portion of the data set.
#     * **Test Data Set** - You are provided the complete test data set; however, you are not given the outcomes.  Your submission is  your predicted outcomes for this data set.
#         * **Public Leaderboard** - You are not told what part of the test data set contributes to the public leaderboard.  Your public score is calculated based on this part of the data set.
#         * **Private Leaderboard** - You are not told what part of the test data set contributes to the public leaderboard.  Your final score/rank is calculated based on this part.  You do not see your private leaderboard score until the end.
# 
# ![How Kaggle Competitions are Scored](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_3_kaggle.png "How Kaggle Competitions are Scored")
# 
# ### Preparing a Kaggle Submission
# 
# Code need not be submitted to Kaggle.  For competitions, you are scored entirely on the accuracy of your sbmission file.  A Kaggle submission file is always a CSV file that contains the **Id** of the row you are predicting and the answer.  For the titanic competition, a submission file looks something like this:
# 
# ```
# PassengerId,Survived
# 892,0
# 893,1
# 894,1
# 895,0
# 896,0
# 897,1
# ...
# ```
# 
# The above file states the prediction for each of various passengers.  You should only predict on ID's that are in the test file.  Likewise, you should render a prediction for every row in the test file.  Some competitions will have different formats for their answers.  For example, a multi-classification will usually have a column for each class and your predictions for each class.

# # Select Kaggle Competitions
# 
# There have been many interesting competitions on Kaggle, these are some of my favorites.
# 
# ## Predictive Modeling
# 
# * [Otto Group Product Classification Challenge](https://www.kaggle.com/c/otto-group-product-classification-challenge)
# * [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
# * [Practice Fusion Diabetes Classification](https://www.kaggle.com/c/pf2012-diabetes)
# * [Predicting a Biological Response](https://www.kaggle.com/c/bioresponse)
# 
# ## Computer Vision
# 
# * [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
# * [Cats vs Dogs](https://www.kaggle.com/c/dogs-vs-cats)
# * [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
# 
# ## Time Series
# 
# * [The Marinexplore and Cornell University Whale Detection Challenge](https://www.kaggle.com/c/whale-detection-challenge)
# 
# ## Other
# 
# * [Helping Santa's Helpers](https://www.kaggle.com/c/helping-santas-helpers)
# 

# In[ ]:





# # Module 8 Assignment
# 
# You can find the first assignment here: [assignment 8](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class8.ipynb)

# In[ ]:




