#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 9: Regularization: L1, L2 and Dropout**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 9 Material
# 
# * Part 9.1: Introduction to Keras Transfer Learning [[Video]](https://www.youtube.com/watch?v=WLlP6S-Z8Xs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_1_keras_transfer.ipynb)
# * Part 9.2: Popular Pretrained Neural Networks for Keras [[Video]](https://www.youtube.com/watch?v=ctVA1_46YEE&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_2_popular_transfer.ipynb)
# * **Part 9.3: Transfer Learning for Computer Vision and Keras** [[Video]](https://www.youtube.com/watch?v=61vMUm_XBMI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_3_transfer_cv.ipynb)
# * Part 9.4: Transfer Learning for Languages and Keras [[Video]](https://www.youtube.com/watch?v=ajmAAg9FxXA&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_4_transfer_nlp.ipynb)
# * Part 9.5: Transfer Learning for Keras Feature Engineering [[Video]](https://www.youtube.com/watch?v=Dttxsm8zpL8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_09_5_transfer_feature_eng.ipynb)

# # Part 9.3: Transfer Learning for Computer Vision and Keras
# 
# In this part we will make use of transfer learning to create a simple neural network that can recognize dog breeds.  To keep the example simple, we will only train for a handfull of breeds.  A much more advanced form of this model can be found at the [Microsoft Dog Breed Image Search](https://www.bing.com/visualsearch/Microsoft/WhatDog).
# 
# To keep computation times to a minimum, we will make use of the MobileNet, which is built into Keras.  We will begin by loading the entire MobileNet and seeing how well it classifies with several test images.  MobileNet can classify 1,000 different images.  We will ultimatly extend it to classify image types that are not in its dataset, in this example 3 dog breeds.  However, we begin by classifying image types amoung those that it was trained on.  Even though our test images were not in its training set, the loaded neural network should be able to classify them.

# In[1]:


import pandas as pd
import numpy as np
import os
import tensorflow.keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# We begin by downloading weights for a MobileNet trained for the imagenet dataset.  This will take some time to download the first time you train the network.

# In[2]:


model = MobileNet(weights='imagenet',include_top=True)


# The loaded network is a Keras neural network, just like those that we've been working with so far.  However, this is a neural network that was trained/engineered on advanced hardware.  Simply looking at the structure of an advanced state-of-the-art neural network can be educational.

# In[3]:


model.summary()


# Just examining the above structure, several clues to neural network architecture become evident.
# 
# Notice how some of the layers have zeros in their number of parameters. Items which are hyperparameters are always zero, nothing about that layer is learned.  The other layers have learnable paramaters that are adjusted as training occurs.  The layer types are all hyperparamaters, Keras will not change a convolution layer to a max pooling layer for you.  However, the layers that have parameters are trained/adjusted by the traning algorithm. Most of the parameters seen above are the weights of the neural network.
# 
# Some of the parameters are maked as non-trainable.  These cannot be adjusted by the training algorithm.  When we later use transfer learning with this model we will strip off the final layers that classify 1000 items and replace with our 3 dog breed classification layer.  Only our new layers will be trainable, we will mark the existing layers as non-trainable.
# 
# The Relu activation function is used throught the neural network.  Also batch and dropout normalization are used.  We cannot see the percent used for batch normalization, that might be specified in the origional paper.  Many deep neural networks are pyramid shaped, and this is the case for this one.  This neural network uses and expanding pyramid shape as you can see the neuron/filter counts expand from 32 to 64 to 128 to 256 to 512 and max out at 1,024.
# 
# We will now use the MobileNet to classify several image URL's below.  You can add additional URL's of your own to see how well the MobileNet can classify.

# In[4]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from io import BytesIO
from IPython.display import display, HTML
from tensorflow.keras.applications.mobilenet import decode_predictions

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

images = [
    "https://cdn.shopify.com/s/files/1/0712/4751/products/SMA-01_2000x.jpg?v=1537468751",
    "https://farm2.static.flickr.com/1394/967537586_87b1358ad3.jpg",
    "https://sites.wustl.edu/jeffheaton/files/2016/07/jheaton_wustl1-262izm5-458x458.jpg",
    "https://1.bp.blogspot.com/-0vGbvWUrSAA/XP-OurPTA4I/AAAAAAAAgtg/TGx6YiGBEGIMjnViDjvVnYzYp__DJ6I-gCLcBGAs/s320/B%252Bt%2525aMbJQkm3Z50rqput%252BA.jpg"
]


def make_square(img):
    cols,rows = img.size
    
    if rows>cols:
        pad = (rows-cols)/2
        img = img.crop((pad,0,cols,cols))
    else:
        pad = (cols-rows)/2
        img = img.crop((0,pad,rows,rows))
    
    return img
        
for url in images:
    x = []
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    
    print("___________________________________________________________________________________________")
    print(img)
    print(np.argmax(pred,axis=1))

    lst = decode_predictions(pred, top=5)
    for itm in lst[0]:
        print(itm)


# Overall, the neural network is doing quite well.  However, it does not classify me as a "person", rather I am classified as a "suit".  Similarly, my English Bulldog Hickory is classiified as a "pug".  This is likely because I am only providiing a closeup of his face.

# For many applications, MobileNet might be entirely acceptable as an image classifier.  However, if you need to classify very specialized images that are not in the 1,000 image types supported by imagenet, it is necessary to use transfer learning.

# ### Transfer
# 
# It is possable to create your own image classification network from scratch.  This would take considerable time and resources.  Just creating a dog breed classifier would require many pictures of dogs, labeled by breed.  By using a pretrained neural network, you are tapping into knowldge already built into the lower layaers of the nerual network.  The transferred layers likely already have some notion of eyes, ears, feet, and fur.  These lower level concepts help to train the neural network to identify dog breeds.
# 
# Next we reload the MobileNet; however, this time we set the *include_top* parameter to *False*. This instructs Keras to not load the final classification layers.  This is the common mode of operation for transfer learning.  We display a summary to see that the top classification layer is now missing.

# In[5]:


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

base_model.summary()


# We will add new top layers to the neural network.  Our final SoftMax layer includes support for 3 classes.

# In[6]:


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) 
x=Dense(1024,activation='relu')(x) 
preds=Dense(3,activation='softmax')(x) 


# Next, we mark the origional MobileNet layers as non-trainable and our new layers as trainable.

# In[7]:


model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True


# To train the neural network we must create a directory structure to hold the images.  The Keras command **flow_from_directory** performs this for us.  It requires that a folder be laid out as follows:
# 
# image
# 
# Each class is a folder that contains images of that class.  We can also specify a target size, in this case the origional MobileNet size of 224x224 is desired.

# In[8]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) 

train_generator=train_datagen.flow_from_directory('c:\\jth\\data\\trans', 
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=1,
                                                 class_mode='categorical',
                                                 shuffle=True)


# We are now ready to compile and fit the neural network.  Notice we are using **fit_generator** rather than **fit**.  This is because we are using the convienent **ImageDataGenerator**.

# In[9]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=50)


# We are now ready to see how our new model can predict dog breeds.  The URLs in the code below provide several example dogs to look at.  Feel free to add your own.

# In[10]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from io import BytesIO
from IPython.display import display, HTML
from tensorflow.keras.applications.mobilenet import decode_predictions

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

images = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/02.Owczarek_niemiecki_u%C5%BCytkowy_kr%C3%B3tkow%C5%82osy_suka.jpg/2560px-02.Owczarek_niemiecki_u%C5%BCytkowy_kr%C3%B3tkow%C5%82osy_suka.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/51/DSHwiki.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Axel%2C_the_English_Bulldog.jpg/440px-Axel%2C_the_English_Bulldog.jpg",
    "https://1.bp.blogspot.com/-0vGbvWUrSAA/XP-OurPTA4I/AAAAAAAAgtg/TGx6YiGBEGIMjnViDjvVnYzYp__DJ6I-gCLcBGAs/s320/B%252Bt%2525aMbJQkm3Z50rqput%252BA.jpg",
    "https://thehappypuppysite.com/wp-content/uploads/2017/12/poodle1.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Pudel_Grossschwarz.jpg/440px-Pudel_Grossschwarz.jpg"
    
]


def make_square(img):
    cols,rows = img.size
    
    if rows>cols:
        pad = (rows-cols)/2
        img = img.crop((pad,0,cols,cols))
    else:
        pad = (cols-rows)/2
        img = img.crop((0,pad,rows,rows))
    
    return img
        
for url in images:
    x = []
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    
    print("___________________________________________________________________________________________")
    print(img)
    print(np.argmax(pred,axis=1))


# In[ ]:




