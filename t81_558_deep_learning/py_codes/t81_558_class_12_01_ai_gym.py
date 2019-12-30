#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 12: Deep Learning and Security**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 12 Video Material
# 
# * **Part 12.1: Introduction to the OpenAI Gym** [[Video]](https://www.youtube.com/watch?v=_KbUxgyisjM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_01_ai_gym.ipynb)
# * Part 12.2: Introduction to Q-Learning [[Video]](https://www.youtube.com/watch?v=uwcXWe_Fra0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_02_qlearningreinforcement.ipynb)
# * Part 12.3: Keras Q-Learning in the OpenAI Gym [[Video]](https://www.youtube.com/watch?v=Ya1gYt63o3M&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_03_keras_reinforce.ipynb)
# * Part 12.4: Atari Games with Keras Neural Networks [[Video]](https://www.youtube.com/watch?v=t2yIu6cRa38&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_04_atari.ipynb)
# * Part 12.5: How Alpha Zero used Reinforcement Learning to Master Chess [[Video]](https://www.youtube.com/watch?v=ikDgyD7nVI8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_05_alpha_zero.ipynb)
# 

# # Part 12.1: Introduction to the OpenAI Gym
# 
# [OpenAI Gym](https://gym.openai.com/) aims to provide an easy-to-setup general-intelligence benchmark with a wide variety of different environments—somewhat akin to, but broader than, the ImageNet Large Scale Visual Recognition Challenge used in supervised learning research—and that hopes to standardize the way in which environments are defined in AI research publications, so that published research becomes more easily reproducible. The project claims to provide the user with a simple interface. As of June 2017, Gym can only be used with Python. As of September 2017, the Gym documentation site was not maintained, and active work focused instead on its GitHub page.
# 
# OpenAI gym is pip-installed onto your local machine.  There are a few important limitations to be aware of:
# 
# * OpenAI Gym Atari only **directly** supports Linux and Macintosh
# * OpenAI Gym Atari can be used with Windows; however, it requires a special [installation procedure](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30)
# * OpenAI Gym needs a graphics display and therefore cannot be **easily** run from a JupyterNotebook environment, such as Google CoLab
# * Because of the additional steps requiired to get OpenAI Gym installed, I am not requiring any assignments based on OpenAI Gym

# ### OpenAI Gym Leaderboard
# 
# The OpenAI Gym does have a leaderboard, similar to Kaggle; however, the OpenAI Gym's leaderboard is much more informal compared to Kaggle.  All scoring is performed on the user's machine.  As a result, the OpenAI gym's leaderboard is strictly an "honor's system".  The leaderboard is maintained the following GitHub repository:
# 
# * [OpenAI Gym Leaderboard](https://github.com/openai/gym/wiki/Leaderboard)
# 
# If you submit a score, you are required to submit a writeup with suffiicient instructions to reproduce your result. A video of your results is suggested, but not required.

# In[ ]:




