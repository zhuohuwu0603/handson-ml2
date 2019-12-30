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
# * **Part 13.3: Using a Keras Deep Neural Network with a Web Application**  [[Video]](https://www.youtube.com/watch?v=OBbw0e-UroI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_03_web.ipynb)
# * Part 13.4: When to Retrain Your Neural Network [[Video]](https://www.youtube.com/watch?v=K2Tjdx_1v9g&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_04_retrain.ipynb)
# * Part 13.5: AI at the Edge: Using Keras on a Mobile Device  [[Video]]() [[Notebook]](t81_558_class_13_05_edge.ipynb)
# 

# # Part 13.3: Using a Keras Deep Neural Network with a Web Application
# 
# In this module we will extend the image API developed in Part 13.1 to work with a web application.  This allows you to use a simple website to upload/predict images, such as this:
# 
# ![MobileNet Web](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/neural-web-1.png "MobileNet Web")
# 
# To do this, we will use the same API developed in Module 13.1.  However, we will now add a [ReactJS](https://reactjs.org/) website around it. This is a single page web application that allows you to upload images for classification by the neural network.  If you would like to read more about ReactJS and image uploading, you can refer to the [blog post](http://www.hartzis.me/react-image-upload/) that I borrowed some of the code from.  I added neural network functionality to a simple ReactJS image upload and preview example.
# 
# This example is built from the following components:
# 
# * [GitHub Location for Web App](./py/)
# * [image_web_server_1.py](./py/image_web_server_1.py) - The code both to start Flask, as well as serve the HTML/JavaScript/CSS needed to provide the web interface.
# * Directory WWW - Contains web assets. 
#     * [index.html](./py/www/index.html) - The main page for the web application.
#     * [style.css](./py/www/style.css) - The stylesheet for the web application.
#     * [script.js](./py/www/script.js) - The JavaScript code for the web application.

# In[ ]:




