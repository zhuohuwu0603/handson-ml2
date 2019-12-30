#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 7: Generative Adversarial Networks**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 7 Material
# 
# * **Part 7.1: Introduction to GANS for Image and Data Generation** [[Video]](https://www.youtube.com/watch?v=0QnCH6tlZgc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_1_gan_intro.ipynb)
# * Part 7.2: Implementing a GAN in Keras [[Video]](https://www.youtube.com/watch?v=T-MCludVNn4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_2_Keras_gan.ipynb)
# * Part 7.3: Face Generation with StyleGAN and Python [[Video]](https://www.youtube.com/watch?v=Wwwyr7cOBlU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_3_style_gan.ipynb)
# * Part 7.4: GANS for Semi-Supervised Learning in Keras [[Video]](https://www.youtube.com/watch?v=ZPewmEu7644&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_4_gan_semi_supervised.ipynb)
# * Part 7.5: An Overview of GAN Research [[Video]](https://www.youtube.com/watch?v=cvCvZKvlvq4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_5_gan_research.ipynb)
# 

# # Part 7.1: Introduction to GANS for Image and Data Generation
# 
# A generative adversarial network (GAN) is a class of machine learning systems invented by Ian Goodfellow in 2014. Two neural networks contest with each other in a game. Given a training set, this technique learns to generate new data with the same statistics as the training set. For example, a GAN trained on photographs can generate new photographs that look at least superficially authentic to human observers, having many realistic characteristics. Though originally proposed as a form of generative model for unsupervised learning, GANs have also proven useful for semi-supervised learning, fully supervised learning, and reinforcement learning.  GANs were introduced in the following paper: 
# 
# * Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). [Generative adversarial nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). In *Advances in neural information processing systems* (pp. 2672-2680).
# 
# This paper used neural networks to automatically generate images for several datasets that we've seen previously: MINST and CIFAR.  However, it also included the Toronto Face Dataset (a private dataset used by some researchers).
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-2.png "GAN")
# 
# Only sub-figure D made use of convolutional neural networks.  Figures A-C make use of fully connected neural networks.  As we will see in this module, the role of convolutional neural networks with GANs was greatly increased.
# 
# A GAN is called a generative model because it generates new data.  The overall process of a GAN is given by the following diagram.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-1.png "GAN")

# In[ ]:





# In[ ]:




