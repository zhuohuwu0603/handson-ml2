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
# * **Part 7.3: Face Generation with StyleGAN and Python** [[Video]](https://www.youtube.com/watch?v=Wwwyr7cOBlU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_3_style_gan.ipynb)
# * Part 7.4: GANS for Semi-Supervised Learning in Keras [[Video]](https://www.youtube.com/watch?v=ZPewmEu7644&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_4_gan_semi_supervised.ipynb)
# * Part 7.5: An Overview of GAN Research [[Video]](https://www.youtube.com/watch?v=cvCvZKvlvq4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_5_gan_research.ipynb)
# 

# # Part 7.3: Face Generation with StyleGAN and Python
# 
# GANs have appeared frequently in the media, showcasing their ability to generate extremely photorealistic faces.  One significant step forward for realistic face generation was nVidia StyleGAN, which was introduced in the following paper.
# 
# * Karras, T., Laine, S., & Aila, T. (2018). [A style-based generator architecture for generative adversarial networks](https://arxiv.org/abs/1812.04948). *arXiv preprint arXiv:1812.04948*.
# 
# In this part we will make use of StyleGAN.  We will also preload weights that nVidia trained on.  This will allow us to generate high resolution photorealistic looking faces, such as this one.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-example246.png "GAN")
# 
# The above image was generated with StyleGAN, using Google CoLab.  Following the instructions in this section, you will be able to create faces like this of your own.  
# 
# While the above image looks much more realistic than the previous set of images, it is not perfect.  There are usually a number of tell-tail signs that you are looking at a computer generated image.  One of the most obvious is usually the surreal, dream-like backgrounds.  The background does not look obviously fake, at first glance; however, upon closer inspection you usually can't quite discern exactly what a GAN generated background actually is.  Also look at the image character's left eye.  It is slightly unrealistic looking, especially near the eyelashes.
# 
# Look at the following GAN face.  Can you spot any imperfections?
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-example221.png "GAN")
# 
# Notice the earrings?  GANs sometimes have problems with symmetry, particularly earrings.

# ### Keras Sequence vs Functional Model API
# 
# Most of the neural networks create in this course have made use of the Keras sequence object.  You might have noticed that we briefly made use of another type of neural network object for the ResNet, the Model.  These are the [two major means](https://keras.io/getting-started/functional-api-guide/) of constructing a neural network in Keras:
# 
# * [Sequential](https://keras.io/getting-started/sequential-model-guide/) - Simplified interface to Keras that supports most models where the flow of information is a simple sequence from input to output. 
# * [Keras Functional API](https://keras.io/getting-started/functional-api-guide/) - More complex interface that allows neural networks to be constructed of reused layers, multiple input layers, and supports building your own recurrent connections.
# 
# It is important to point out that these are not two specific types of neural network.  Rather, they are two means of constructing neural networks in Keras.  Some types of neural network can be implemented in either, such as dense feedforward neural networks (like we used for the Iris and MPG datasets).  However, other types of neural network, like ResNet and GANs can only be used in the Functional Model API.

# ### Generating High Rez GAN Faces with Google CoLab
# 
# This notebook demonstrates how to run [NVidia StyleGAN](https://github.com/NVlabs/stylegan) inside of a Google CoLab notebook.  I suggest you use this to generate GAN faces from a pretrained model.  If you try to train your own, you will run into compute limitations of Google CoLab.
# 
# Make sure to run this code on a GPU instance.  GPU is assumed.
# 
# First, map your G-Drive, this is where your GANs will be written to.

# In[ ]:


# Run this for Google CoLab
from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# Next, clone StyleGAN from GitHub.

# In[ ]:


# get_ipython().system('git clone https://github.com/NVlabs/stylegan.git')


# Verify that StyleGAN has been cloned.

# In[ ]:


# get_ipython().system('ls /content/stylegan/')


# Add the StyleGAN folder to Python so that you can import it.

# In[ ]:


import sys
sys.path.insert(0, "/content/stylegan")

import dnnlib


# The code below is based on code from NVidia. This actually generates your images.

# In[ ]:


# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState()
    

    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'/content/drive/My Drive/images/example1.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()


# In[ ]:




