#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 13: Advanced/Other Topics**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 13 Video Material
# 
# * Part 13.1: Flask and Deep Learning Web Services [[Video]](https://www.youtube.com/watch?v=H73m9XvKHug&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_01_flask.ipynb)
# * Part 13.2: Deploying a Model to AWS  [[Video]](https://www.youtube.com/watch?v=8ygCyvRZ074&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_02_cloud.ipynb)
# * Part 13.3: Using a Keras Deep Neural Network with a Web Application  [[Video]](https://www.youtube.com/watch?v=OBbw0e-UroI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_03_web.ipynb)
# * Part 13.4: When to Retrain Your Neural Network [[Video]](https://www.youtube.com/watch?v=K2Tjdx_1v9g&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_04_retrain.ipynb)
# * **Part 13.5: AI at the Edge: Using Keras on a Mobile Device**  [[Video]]() [[Notebook]](t81_558_class_13_05_edge.ipynb)
# 

# # Part 13.5: Using a Keras Deep Neural Network with a Web Application
# 
# In this part we will see how to deploy a neural network to an iOS mobile device.  Android is also another option that I plan to support at some point.  However, for now, I am focusing on iOS.  Apple added their [CoreML](https://developer.apple.com/documentation/coreml) library that makes it considerably easier to deploy a deep neural network than it used to be. The example in this part will focus on creating a simple computer vision mobile application for image recognition.  All computation will occur on the actual device.  This is called computing on the "edge", as opposed to "cloud compute".
# 
# Apple makes [several pre-trained](https://developer.apple.com/machine-learning/models/) neural networks available for CoreML.  It is also possible to convert Keras models into the format needed by CoreML.  For this example we will convert a Keras pre-trained model to CoreML.  This gives a good demonstration of this conversion that can use for other Keras models that you've created. 
# 
# Please note the following two requirements set forth by Apple.
# 
# * You will need a Mac running [XCode]() to create an iOS application.
# * You must have a free Apple Developer account to deploy your app to your own iOS device.  [Sign up here](https://developer.apple.com/).
# * To add your application to the [Apple App Store](https://www.apple.com/ios/app-store/) and deploy to other peoples hardware, you mist [enroll](https://developer.apple.com/support/compare-memberships/) in the $100/year developer program.
# 
# ### Converting Keras to CoreML
# 
# The following code exports MobileNet to an H5 file. 

# ```
# conda create -y --name coreml python=3.6
# source activate coreml
# conda install -y jupyter
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
# pip install --exists-action i --upgrade tensorflow==1.14
# pip install --exists-action i --upgrade keras==2.2.4
# pip install --exists-action i --upgrade coremltools
# conda update -y --all
# python -m ipykernel install --user --name coreml --display-name "Python 3.6 (coreml)"
# ```

# In[7]:


# Export MobileNet to an H5 file
import os
from keras.applications import MobileNet

save_path = "./dnn/"
model = MobileNet(weights='imagenet',include_top=True)
model.save(os.path.join(save_path,"mobilenet.h5"))


# In[ ]:


Unfortunately, as of August 2019, CoreML does not support TensorFlow 2.0. 


# In[8]:


import tensorflow as tf
import keras
import coremltools

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")


# In[31]:


import requests
r = requests.get('https://data.heatonresearch.com/data/t81-558/imagenet_class_index.json')

js = r.json()

lookup = ['' for x in range(1000)]
for idx in js:
    lookup[int(idx)] = js[idx][1]


# In[32]:


coreml_model = coremltools.converters.keras.convert(model,
    input_names="image",
    image_input_names="image",
    image_scale=1/255.0,
    class_labels=lookup,
    is_bgr=True)


# In[33]:


coreml_model.save(os.path.join(save_path,"mobilenet.mlmodel"))


# ### Creating an IOS CoreML Application
# 
# We will now use the neural network created in the last section to create an IOS application that will classify what its camera sees.  This will be a single image classification, not the multi-image classification that we saw with YOLO.  You can see the application running on my iPhone here:
# 
# ![IOS Image Classify](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/ios-1.png "IOS Image Classify")
# 
# The complete source code (in XCode) for this application can be found at the following URL:
# 
# * [GitHub: IOS Classify](https://github.com/jeffheaton/ios_video_classify)
# 
# To create this application from scratch (in XCode), follow these steps:
# 
# 1. Install XCode
# 2. Register for Apple Developer account (if you wish to deploy to iOS device)
# 3. Create new XCode Project
# 4. Delete storyboard
# 5. Remove project references to storyboard
# 6. Add camera prompt to security settings
# 7. Replace the contents of the view controller with the included
# 8. Test on IOS device
# 
# The YouTube video for this module goes through the above process.

# In[ ]:





# ### More Reading
# 
# There are a number of very good tutorials on IOS and CoreML development.  The following articles were very helpful in the creation of this material.
# 
# * [Running Keras models on iOS with CoreML](https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)
# * [How to build an image recognition iOS app with Apple’s CoreML and Vision APIs](https://www.freecodecamp.org/news/ios-coreml-vision-image-recognition-3619cf319d0b/)

# In[ ]:




