#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 6: Convolutional Neural Networks (CNN) for Computer Vision**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 6 Material
# 
# * Part 6.1: Image Processing in Python [[Video]](https://www.youtube.com/watch?v=4Bh3gqHkIgc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_06_1_python_images.ipynb)
# * Part 6.2: Keras Neural Networks for Digits and Fashion MNIST [[Video]](https://www.youtube.com/watch?v=-SA8BmGvWYE&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_06_2_cnn.ipynb)
# * Part 6.3: Implementing a ResNet in Keras** [[Video]](https://www.youtube.com/watch?v=qMFKsMeE6fM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_06_3_resnet.ipynb)
# * **Part 6.4: Using Your Own Images with Keras** [[Video]](https://www.youtube.com/watch?v=VcFja1fUNSk&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_06_4_keras_images.ipynb)
# * Part 6.5: Recognizing Multiple Images with YOLO Darknet [[Video]](https://www.youtube.com/watch?v=oQcAKvBFli8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_06_5_yolo.ipynb)

# In[1]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


# # Part 6.4: Using Your Own Images with Keras
# 
# So far we've used image data sets that Keras provides convenience functions for accessing.  There are a number of [built-in data sets](https://www.tensorflow.org/api_docs/python/tf/keras/datasets) for Keras.  While these convenience functions do make it easier to create Keras models for these data sets, these functions also hide the internal workings.  You might be wondering how you would train a neural network from your own sets of images.  
# 
# Consider the convenience functions provided for CIFAR-10:

# In[2]:


from tensorflow.keras.datasets import cifar10
import numpy as np

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# The above code extracts the training and test sets from CIFAR-10.  Often these datasets are already pre-split between test and training data.  This allows comparison between many researchers who are working on models for this data.  Without these splits, it would be difficult to compare accuracy results between two different researchers that were using two different train/test splits.  Consider the shape of the dataset.

# In[3]:


x_train.shape


# We are provided with 50,000 training elements.  Each training element is an image that is 32x32 pixels with 3 color channels.  Typically you will either see 1 color channel (grayscale) or 3 color channels (RGB color). 
# 
# If we look inside of one of the 50,000 elements we can see the structure of each image.  It is a matrix of RGB values.

# In[4]:


x_train[0]


# It is also important to note that the data type is uint8, which is unsigned integer 8-bits (1 byte).  This corresponds well with image binary data (that is typically 24-bit, 8 bits per 3 color channel = 24 bit).  However, while the images may be 8-bit based, neural networks typically expect floating point input.  Because of this, some transformation/normalization of the data is needed.
# 
# When training, it is usually necessary to handle multiple images at a time.  The code presented here will load in multiple images and convert them so that they are all the same size.  Processing training data images so that each image is of a uniform height and width is a very common step for computer vision programs.

# In[5]:


training_data = []

# get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image, ImageFile
from matplotlib.pyplot import imshow
import requests
import numpy as np
from io import BytesIO
from IPython.display import display, HTML

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_CHANNELS = 3

images = [
    "https://upload.wikimedia.org/wikipedia/commons/9/92/Brookings.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/ff/WashU_Graham_Chapel.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/9/9e/SeigleHall.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/aa/WUSTLKnight.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/3/32/WashUABhall.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c0/Brown_Hall.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f4/South40.jpg"    
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
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.load()
    img = make_square(img)
    img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
    training_data.append(np.asarray(img))


# The above code contains a function called **make_square** that ensures that each of the images have the same height and width.  Square images are particularly easy to deal with because each image will have the same aspect ratio.  There are several techniques that can be used to make an image square.  Fundamentally the image will either be cropped or padded to make it square. Padding adds extra space to the image to cause a square shape.  Cropping removes pixels (and therefore information) from the images to make them square. 
# 
# The technique above crops the images to make them square.  The row and column sizes are analyzed and the image is adjusted based on if the row or column size is smaller.  If there are fewer rows and than columns then the extra columns are dropped to cause the row and column count to be equal. Similarly, if there are fewer columns and than rows then the extra columns are rows to cause the row and column count to be equal. 
# 
# For each image in the set, the image is first adjusted to be square and then resized to the common size that we are forcing all images to be.  The resulting images are each added to a list.  The training data, at this point, is shown below.

# In[6]:


training_data


# We now have a 1D list of 3D images (height by width by color depth).  We would like this to be a 4D Numpy array.  Passing the list to **np.array** performs this conversion/reshaping.
# 
# The training data is divided by 127.5 and subtracted by one to normalize to between -1 and 1.  This causes the RGB values to be centered around zero and gives greater predictive power to the neural network.

# In[7]:


training_data = np.array(training_data) / 127.5 - 1.


# We can display the normalized training data:

# In[8]:


training_data


# It is sometimes useful to to save a training set.  For image and higher dimensional data, as CSV file is not sufficient. Also, Pickle can experience problems with very large datasets.  Because of this I prefer to use Numpy's own format for binary data. 

# In[9]:


print("Saving training image binary...")
np.save("training",training_data) # Saves as "training.npy"
print("Done.")

