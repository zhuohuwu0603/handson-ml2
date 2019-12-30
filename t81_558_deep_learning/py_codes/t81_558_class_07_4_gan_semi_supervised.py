#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 7: Generative Adversarial Networks**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 7 Material
# 
# * Part 7.1: Introduction to GANS for Image and Data Generation [[Video]](https://www.youtube.com/watch?v=0QnCH6tlZgc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_1_gan_intro.ipynb)
# * Part 7.2: Implementing a GAN in Keras [[Video]](https://www.youtube.com/watch?v=T-MCludVNn4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_2_Keras_gan.ipynb)
# * Part 7.3: Face Generation with StyleGAN and Python [[Video]](https://www.youtube.com/watch?v=Wwwyr7cOBlU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_3_style_gan.ipynb)
# * **Part 7.4: GANS for Semi-Supervised Learning in Keras** [[Video]](https://www.youtube.com/watch?v=ZPewmEu7644&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_4_gan_semi_supervised.ipynb)
# * Part 7.5: An Overview of GAN Research [[Video]](https://www.youtube.com/watch?v=cvCvZKvlvq4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_5_gan_research.ipynb)
# 
# 

# # Part 7.4: GANS for Semi-Supervised Training in Keras
# 
# GANs can also be used to implement semi-supervised learning/training.  Normally GANs implement un-supervised training.  This is because there are no y's (expected outcomes) provided in the dataset.  The y-values are usually called labels.  For the face generating GANs, there is typically no y-value, only images.  This is unsupervised training.  Supervised training occurs when we are training a model to 
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-training.png "GAN")
# 
# The following paper describes the application of GANs to semi-supervised training.
# 
# * [Odena, A. (2016). Semi-supervised learning with generative adversarial networks. *arXiv preprint* arXiv:1606.01583.](https://arxiv.org/abs/1606.01583)
# 
# As you can see, supervised learning is where all data have labels.  Supervised learning attempts to learn the labels from the training data to predict these labels for new data.  Un-supervised learning has no labels and usually simply clusters the data or in the case of a GAN, learns to produce new data that resembles the training data.  Semi-supervised training has a small number of labels for mostly unlabeled data.  Semi-supervised learning is usually similar to supervised learning in that the goal is ultimately to predict labels for new data.
# 
# Traditionally, unlabeled data would simply be discarded if the overall goal was to create a supervised model.  However, the unlabeled data is not without value.  Semi-supervised training attempts to use this unlabeled data to help learn additional insights about what labels we do have.  There are limits, however.  Even semi-supervised training cannot learn entirely new labels that were not in the training set.  This would include new classes for classification or learning to predict values outside of the range of the y-values.
# 
# Semi-supervised GANs can perform either classification or regression.  Previously, we made use of the generator and discarded the discriminator.  We simply wanted to create new photo-realistic faces, so we just needed the generator.  Semi-supervised learning flips this, as we now discard the generator and make use of the discriminator as our final model.
# 
# ### Semi-Supervised Classification Training
# 
# The following diagram shows how to apply GANs for semi-supervised classification training.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-semi-class.png "GAN")
# 
# Semi-supervised classification training is laid exactly the same as a regular GAN.  The only differences is that it is not a simple true/false classifier as was the case for image GANs that simply classified if the generated image was a real or fake.  The additional classes are also added.  Later in this module I will provide a link to an example of [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/).  This dataset contains house numbers, as seen in the following image.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/svhn_examples_new.png "GAN")
# 
# Perhaps all of the digits are not labeled.  The GAN is setup to classify a real or fake digit, just as we did with the faces.  However, we also expand upon the real digits to include classes 0-9.  The GAN discriminator is classifying between the 0-9 digits and also fake digits.  A semi-supervised GAN classifier always classifies to the number of classes plus one. The additional class indicates a fake classification. 
# 
# ### Semi-Supervised Regression Training
# 
# The following diagram shows how to apply GANs for semi-supervised regression training.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-semi-reg.png "GAN")
# 
# Neural networks can perform both classification and regression simultaneously, it is simply a matter of how the output neurons are mapped.  A hybrid classification-regression neural network simply maps groups of output neurons to be each of the groups of classes to be predicted, along with individual neurons to perform any regression predictions needed.
# 
# A regression semi-supervised GAN is one such hybrid.  The discriminator has two output neurons.  The first output neuron performs the requested regression prediction.  The second predicts the probability that the input was fake.
# 
# ### Application of Semi-Supervised Regression
# 
# An example of using Keras for Semi-Supervised classification is provided here.
# 
# * [Semi-supervised learning with Generative Adversarial Networks (GANs)](https://towardsdatascience.com/semi-supervised-learning-with-gans-9f3cb128c5e)
# * [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
# * [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

# In[ ]:




