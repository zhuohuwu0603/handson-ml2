#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_2_Keras_gan.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # T81-558: Applications of Deep Neural Networks
# **Module 7: Generative Adversarial Networks**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 7 Material
# 
# * Part 7.1: Introduction to GANS for Image and Data Generation [[Video]](https://www.youtube.com/watch?v=0QnCH6tlZgc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_1_gan_intro.ipynb)
# * **Part 7.2: Implementing a GAN in Keras** [[Video]](https://www.youtube.com/watch?v=T-MCludVNn4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_2_Keras_gan.ipynb)
# * Part 7.3: Face Generation with StyleGAN and Python [[Video]](https://www.youtube.com/watch?v=Wwwyr7cOBlU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_3_style_gan.ipynb)
# * Part 7.4: GANS for Semi-Supervised Learning in Keras [[Video]](https://www.youtube.com/watch?v=ZPewmEu7644&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_4_gan_semi_supervised.ipynb)
# * Part 7.5: An Overview of GAN Research [[Video]](https://www.youtube.com/watch?v=cvCvZKvlvq4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_07_5_gan_research.ipynb)
# 

# In[ ]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# # Part 7.2: Implementing DCGANs in Keras
# 
# Paper that described the type of DCGAN that we will create in this module.
# 
# * Radford, A., Metz, L., & Chintala, S. (2015). [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/abs/1511.06434). *arXiv preprint arXiv:1511.06434*.
# 
# This paper implements a DCGAN as follows:
# 
# * No pre-processing was applied to training images besides scaling to the range of the tanh activation function [-1, 1]. 
# * All models were trained with mini-batch stochastic gradient descent (SGD) with a mini-batch size of 128. 
# * All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02. 
# * In the LeakyReLU, the slope of the leak was set to 0.2 in all models.
# * we used the Adam optimizer(Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested learning rate of 0.001, to be too high, using 0.0002 instead. 
# * Additionally, we found leaving the momentum term $\beta{1}$ at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.
# 
# The paper also provides the following architecture guidelines for stable Deep Convolutional GANs:
# 
# * Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
# * Use batchnorm in both the generator and the discriminator.
# * Remove fully connected hidden layers for deeper architectures.
# * Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# * Use LeakyReLU activation in the discriminator for all layers.
# 
# While creating the material for this module I used a number of Internet resources, some of the most helpful were:
# 
# * [Deep Convolutional Generative Adversarial Network (TensorFlow 2.0 example code)](https://www.tensorflow.org/tutorials/generative/dcgan)
# * [Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)
# * [Collection of Keras implementations of Generative Adversarial Networks GANs](https://github.com/eriklindernoren/Keras-GAN)
# * [dcgan-facegenerator](https://github.com/platonovsimeon/dcgan-facegenerator), [Semi-Paywalled Article by GitHub Author](https://medium.com/datadriveninvestor/generating-human-faces-with-keras-3ccd54c17f16)
# 
# The program created next will generate faces similar to these.  While these faces are not perfect, they demonstrate how we can construct and train a GAN on or own.  Later we will see how to import very advanced weights from nVidia to produce high resolution, realistic looking faces.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan-3.png "GAN Images")
# 
# As discussed in the previous module, the GAN is made up of two different neural networks: the discriminator and the generator.  The generator generates the images, while the discriminator detects if a face is real or was generated.  These two neural networks work as follows:
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan_fig_1.png "GAN")
# 
# The discriminator accepts an image as its input and produces number that is the probability of the input image being real.  The generator accepts a random seed vector and generates an image from that random vector seed. An unlimited number of new images can be created by providing additional seeds.

# I suggest running this code with a GPU, it will be very slow on a CPU alone.  The following code mounts your Google drive for use with Google CoLab.  If you are not using CoLab, the following code will not work.

# In[2]:



try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
    # get_ipython().run_line_magic('tensorflow_version', '2.x')
except:
    print("Note: not using Google CoLab")
    COLAB = True


# The following packages will be used to implement a basic GAN system in Python/Keras.

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt


# These are the constants that define how the GANs will be created for this example.  The higher the resolution, the more memory that will be needed.  Higher resolution will also result in longer run times.  For Google CoLab (with GPU) 128x128 resolution is as high as can be used (due to memory).  Note that the resolution is specified as a multiple of 32.  So **GENERATE_RES** of 1 is 32, 2 is 64, etc.
# 
# To run this you will need training data.  The training data can be any collection of images.  I suggest using training data from the following two locations.  Simply unzip and combine to a common directory.  This directory should be uploaded to Google Drive (if you are using CoLab). The constant **DATA_PATH** defines where these images are stored.
# 
# The source data (faces) used in this module can be found here:
# 
# * [Kaggle Faces Data New](https://www.kaggle.com/gasgallo/faces-data-new)
# * [Kaggle Lag Dataset: Dataset of faces, from more than 1k different subjects](https://www.kaggle.com/gasgallo/lag-dataset)

# In[4]:


# Generation resolution - Must be square 
# Training data is also scaled to this.
# Note GENERATE_RES 4 or higher  will blow Google CoLab's memory and have not
# been tested extensivly.
GENERATE_RES = 3 # Generation resolution factor (1=32, 2=64, 3=96, 4=128, etc.)
GENERATE_SQUARE = 32 * GENERATE_RES # rows/cols (should be square)
IMAGE_CHANNELS = 3

# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
DATA_PATH = '/content/drive/My Drive/projects/faces'
EPOCHS = 50
BATCH_SIZE = 32
BUFFER_SIZE = 60000

print(f"Will generate {GENERATE_SQUARE}px square images.")


# Next we will load and preprocess the images.  This can take awhile.  Google CoLab took around an hour to process.  Because of this we store the processed file as a binary.  This way we can simply reload the processed training data and quickly use it.  It is most efficient to only perform this operation once.  The dimensions of the image are encoded into the filename of the binary file because we need to regenerate it if these change.

# In[5]:


# Image set has 11,682 images.  Can take over an hour for initial preprocessing.
# Because of this time needed, save a Numpy preprocessed file.
# Note, that file is large enough to cause problems for sume verisons of Pickle,
# so Numpy binary files are used.
training_binary_path = os.path.join(DATA_PATH,f'training_data_{GENERATE_SQUARE}_{GENERATE_SQUARE}.npy')

print(f"Looking for file: {training_binary_path}")

if not os.path.isfile(training_binary_path):
  start = time.time()
  print("Loading training images...")

  training_data = []
  faces_path = os.path.join(DATA_PATH,'face_images')
  for filename in tqdm(os.listdir(faces_path)):
      path = os.path.join(faces_path,filename)
      image = Image.open(path).resize((GENERATE_SQUARE,GENERATE_SQUARE),Image.ANTIALIAS)
      training_data.append(np.asarray(image))
  training_data = np.reshape(training_data,(-1,GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS))
  training_data = training_data.astype(np.float32)
  training_data = training_data / 127.5 - 1.


  print("Saving training image binary...")
  np.save(training_binary_path,training_data)
  elapsed = time.time()-start
  print (f'Image preprocess time: {hms_string(elapsed)}')
else:
  print("Loading previous training pickle...")
  training_data = np.load(training_binary_path)


# We will use a TensorFlow **Dataset** object to actually hold the images.  This allows the data to be quickly shuffled int divided into the appropriate batch sizes for training.  

# In[ ]:


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# The code below creates the generator and discriminator.

# Next we actually build the discriminator and the generator.  Both will be trained with the Adam optimizer.

# In[ ]:


def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4*4*256,activation="relu",input_dim=seed_size))
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if GENERATE_RES>1:
      model.add(UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))
      model.add(Conv2D(128,kernel_size=3,padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    return model


def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model


# As we progress through training images will be produced to show the progress.  These images will contain a number of rendered faces that show how good the generator has become.  These faces will be 

# In[ ]:


def save_images(cnt,noise):
  image_array = np.full(( 
      PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 
      PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE+PREVIEW_MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generated_images = generator.predict(noise)

  generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        c = col * (GENERATE_SQUARE+16) + PREVIEW_MARGIN
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)


# 

# In[9]:


generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)

noise = tf.random.normal([1, SEED_SIZE])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])


# 

# In[10]:


image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)
decision = discriminator(generated_image)
print (decision)


# Loss functions must be developed that allow the generator and discriminator to be trained in an adversarial way.  Because these two neural networks are being trained independently they must be trained in two separate passes.  This requires two separate loss functions and also two separate updates to the gradients.  When the discriminator's gradients are applied to decrease the discriminator's loss it is important that only the discriminator's weights are update.  It is not fair, nor will it produce good results, to adversarially damage the weights of the generator to help the discriminator.  A simple backpropagation would do this.  It would simultaneously affect the weights of both generator and discriminator to lower whatever loss it was assigned to lower.
# 
# The following diagram shows how the discriminator is trained.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan_fig_2.png "GAN")
# 
# Here a training set is generated with an equal number of real and fake images.  The real images are randomly sampled (chosen) from the training data.  An equal number of random images are generated from random seeds.  For the discriminator training set, the $x$ contains the input images and the $y$ contains a value of 1 for real images and 0 for generated ones.
# 
# Likewise, the following diagram shows how the generator is trained.
# 
# ![GAN](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/gan_fig_3.png "GAN")
# 
# For the generator training set, the $x$ contains the random seeds to generate images and the $y$ always contains the value of 1, because the optimal is for the generator to have generated such good images that the discriminiator was fooled into assigning them a probability near 1.

# In[ ]:


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Both the generator and discriminator use Adam and the same learning rate and momentum.  This does not need to be the case.  If you use a **GENERATE_RES** greater than 3 you may need to tune these learning rates, as well as other training and hyperparameters.  

# In[ ]:


generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)


# The following function is where most of the training takes place for both the discriminator and the generator.  This function was based on the GAN provided by the [TensorFlow Keras exmples](https://www.tensorflow.org/tutorials/generative/dcgan) documentation.  The first thing you should notice about this function is that it is annotated with the **tf.function** annotation.  This causes the function to be precompiled and improves performance.
# 
# This function trans differently than the code we previously saw for training.  This code makes use of **GradientTape** to allow the discriminator and generator to be trained together, yet separately.  
# 
# 

# In[ ]:


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  return gen_loss,disc_loss


# In[ ]:


def train(dataset, epochs):
  fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    epoch_start = time.time()

    gen_loss_list = []
    disc_loss_list = []

    for image_batch in dataset:
      t = train_step(image_batch)
      gen_loss_list.append(t[0])
      disc_loss_list.append(t[1])

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    epoch_elapsed = time.time()-epoch_start
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {hms_string(epoch_elapsed)}')
    save_images(epoch,fixed_seed)

  elapsed = time.time()-start
  print (f'Training time: {hms_string(elapsed)}')


# In[15]:


train(train_dataset, EPOCHS)


# 

# In[ ]:


generator.save(os.path.join(DATA_PATH,"face_generator.h5"))

