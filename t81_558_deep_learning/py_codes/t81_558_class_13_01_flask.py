#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 13: Advanced/Other Topics**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 13 Video Material
# 
# * **Part 13.1: Flask and Deep Learning Web Services** [[Video]](https://www.youtube.com/watch?v=H73m9XvKHug&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_01_flask.ipynb)
# * Part 13.2: Deploying a Model to AWS  [[Video]](https://www.youtube.com/watch?v=8ygCyvRZ074&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_02_cloud.ipynb)
# * Part 13.3: Using a Keras Deep Neural Network with a Web Application  [[Video]](https://www.youtube.com/watch?v=OBbw0e-UroI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_03_web.ipynb)
# * Part 13.4: When to Retrain Your Neural Network [[Video]](https://www.youtube.com/watch?v=K2Tjdx_1v9g&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_04_retrain.ipynb)
# * Part 13.5: AI at the Edge: Using Keras on a Mobile Device  [[Video]]() [[Notebook]](t81_558_class_13_05_edge.ipynb)
# 

# # Part 13.1: Flask and Deep Learning Web Services
# 
# If your neural networks are to be woven into a production system they must be exposed in a way that they can be easily executed by Python and other programming languages.  The usual means for doing this is a web service. One of the most popular libraries for doing this in Python is [Flask](https://palletsprojects.com/p/flask/). This library allows you to quickly deploy your Python applications, including TensorFlow, as web services.
# 
# Deployment is a complex process, usually carried out by a company's [Information Technology (IT) group](https://en.wikipedia.org/wiki/Information_technology).  When large numbers of clients must access your model scalability becomes important.  This is usually handled by the cloud.  Flask is not designed for high-volume systems.  When deployed to production, models will usually be wrapped in [Gunicorn](https://gunicorn.org/) or TensorFlow Serving.  High volume cloud deployment is discussed in the next chapter.  Everything presented in this part ith Flask is directly compatible with the higher volume Gunicorn system. It is common to use a development system, such as Flask, when you are developing your initial system.
# 
# ### Other Material
# 
# The following articles might also be useful for greater understanding of Flask.
# 
# * [Flask Quickstart](https://flask.palletsprojects.com/en/1.0.x/quickstart/)
# 
# ### Flask Hello World
# 
# It is uncommon to run Flask from a Jupyter notebook.  Flask is the server and Jupyter usually fills the role of the client.  However, we can run a simple web service from Jupyter.  We will quickly move beyond this and deploy using a Python script (.py).  This means that it will be difficult to use Google CoLab, as you will be running from the command line.  For now, lets execute a Flask web container in Jupyter.

# In[1]:


from werkzeug.wrappers import Request, Response
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)


# This program simply starts a web service on port 9000 of your computer.  This cell will remain running (appearing locked up).  However, it is simply waiting for browsers to connect.  If you point your browser at the following URL, you will interact with the Flask web service.
# 
# * http://localhost:9000/
# 
# You should see Hello World displayed.

# ### MPG Flask
# 
# Usually you will interact with a web service through JSON.  A JSON message will be sent to your Flask application and a JSON response will be returned.  Later, in module 13.3, we will see how to attach this web service to a web application that you can interact with through a browser.  We will create a Flask wrapper for a neural network that predicts the miles per gallon for a car.  The sample JSON will look like this.
# 
# ```
# {
#   "cylinders": 8, 
#   "displacement": 300,
#   "horsepower": 78, 
#   "weight": 3500,
#   "acceleration": 20, 
#   "year": 76,
#   "origin": 1
# }
# ```
# 
# We will see two different means of POSTing this JSON data to our web server.  First, we will use a utility called [POSTman](https://www.getpostman.com/).  Secondly, we will use Python code to construct the JSON message and interact with Flask. 
# 
# First, it is necessary to train a neural network with the MPG dataset.  This is very similar to what we've done many times before.  However, we will save the neural network so that we can load it later.  We do not want to have Flask actually train the neural network.  We wish to have the neural network already trained and simply deploy the already trained .H5 file that we save the neural network to.  The following code trains a MPG neural network.

# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])

cars = df['name']

# Handle missing value
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Pandas to Numpy
x = df[['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'year', 'origin']].values
y = df['mpg'].values # regression

# Split into validation and training sets
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.25, random_state=42)

# Build the neural network
model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
model.add(Dense(10, activation='relu')) # Hidden 2
model.add(Dense(1)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto',
        restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),callbacks=[monitor],verbose=2,epochs=1000)


# Next we evaluate the score.  This is more of a sanity check to ensure the code above worked as expected. 

# In[3]:


pred = model.predict(x_test)
# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"After load score (RMSE): {score}")


# Next we save the neural network to a .H5 file.

# In[4]:


model.save(os.path.join("./dnn/","mpg_model.h5"))


# We would like the Flask web service to check that the input JSON is valid.  To do this, we need to know what values we expect and what their logical ranges are.  The following code outputs the expected fields, their ranges, and packages all of this information into a JSON object that is copied to the Flask web application.  This will allow us to validate the incoming JSON requests.

# In[5]:


cols = [x for x in df.columns if x not in ('mpg','name')]

print("{")
for i,name in enumerate(cols):
    print(f'"{name}":{{"min":{df[name].min()},"max":{df[name].max()}}}{"," if i<(len(cols)-1) else ""}')
print("}")


# Finally, we setup Python code that will call the model for a single car and get a prediction.  This code will also be copied to the Flask web application.

# In[6]:


import os
from tensorflow.keras.models import load_model
import numpy as np

model = load_model(os.path.join("./dnn/","mpg_model.h5"))
x = np.zeros( (1,7) )

x[0,0] = 8 # 'cylinders', 
x[0,1] = 400 # 'displacement', 
x[0,2] = 80 # 'horsepower', 
x[0,3] = 2000 # 'weight',
x[0,4] = 19 # 'acceleration', 
x[0,5] = 72 # 'year', 
x[0,6] = 1 # 'origin'


pred = model.predict(x)
float(pred[0])


# The completed web application can be found here:
#     
# * [mpg_server_1.py](./py/mpg_server_1.py)
# 
# This server can be run from the command line with:
# 
# ```
# python mpg_server_1.py
# ```
# 
# If you are using a virtual environment (described in Module 1.1), make sure to use the ```activate tensorflow``` command for Windows or ```source activate tensorflow``` for Mac before executing the above command.

# ### Flask MPG Client
# 
# Now that we have a web service running we would like to access it.  This is a bit more complex than the "Hello World" web server we first saw in this part.  The request to display was an HTTP GET.  We must now do an HTTP POST.  To accomplish this you must use a client.  We will see both how to use [PostMan](https://www.getpostman.com/), as well as directly through a Python program in Jupyter.
# 
# We will begin with PostMan.  If you have not already done so, install PostMan.  
# 
# To successfully use PostMan to query your web service you must enter the following settings:
# 
# * POST Request to http://localhost:5000/api/mpg
# * RAW JSON and paste in JSON from above
# * Click Send and you should get a correct result
# 
# The following shows a successful result.
# 
# ![PostMan JSON](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/postman-1.png "PostMan JSON")
# 
# This same process can be done programmatically in Python.

# In[7]:


import requests

json = {
  "cylinders": 8, 
  "displacement": 300,
  "horsepower": 78, 
  "weight": 3500,
  "acceleration": 20, 
  "year": 76,
  "origin": 1
}

r = requests.post("http://localhost:5000/api/mpg",json=json)
if r.status_code == 200:
    print("Success: {}".format(r.text))
else: print("Failure: {}".format(r.text))


# ### Images and Web Services
# 
# We can also accept images from web services.  We will create web service that accepts images and classifies them using MobileNet.  To use your own neural network you will follow the same process just load your own network like we did for the MPG example. The completed web service can be found here:
# 
# [image_server_1.py](./py/image_server_1.py)
# 
# This server can be run from the command line with:
# 
# ```
# python mpg_server_1.py
# ```
# 
# If you are using a virtual environment (described in Module 1.1), make sure to use the ```activate tensorflow``` command for Windows or ```source activate tensorflow``` for Mac before executing the above command.
# 
# To successfully use PostMan to query your web service you must enter the following settings:
# 
# * POST Request to http://localhost:5000/api/image
# * Use "Form Data" and create one entry named "image" that is a file.  Choose an image file to classify.
# * Click Send and you should get a correct result
# 
# The following shows a successful result.
# 
# ![PostMan Image](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/postman-2.png "PostMan Image")
# 
# This same process can be done programmatically in Python.

# In[8]:


import requests
response = requests.post('http://localhost:5000/api/image', files=dict(image=('hickory.jpeg',open('./photos/hickory.jpeg','rb'))))
if response.status_code == 200:
    print("Success: {}".format(response.text))
else: print("Failure: {}".format(response.text))


# In[ ]:





# In[ ]:




