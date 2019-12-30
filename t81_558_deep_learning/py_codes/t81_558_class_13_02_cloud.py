#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 13: Advanced/Other Topics**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 13 Video Material
# 
# * Part 13.1: Flask and Deep Learning Web Services [[Video]](https://www.youtube.com/watch?v=H73m9XvKHug&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_01_flask.ipynb)
# * **Part 13.2: Deploying a Model to AWS**  [[Video]](https://www.youtube.com/watch?v=8ygCyvRZ074&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_02_cloud.ipynb)
# * Part 13.3: Using a Keras Deep Neural Network with a Web Application  [[Video]](https://www.youtube.com/watch?v=OBbw0e-UroI&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_03_web.ipynb)
# * Part 13.4: When to Retrain Your Neural Network [[Video]](https://www.youtube.com/watch?v=K2Tjdx_1v9g&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_13_04_retrain.ipynb)
# * Part 13.5: AI at the Edge: Using Keras on a Mobile Device  [[Video]]() [[Notebook]](t81_558_class_13_05_edge.ipynb)
# 

# # Part 13.2: Deploying a Model to AWS
# 
# Some additional material:
# 
# * [Serving TensorFlow Models](https://www.tensorflow.org/tfx/guide/serving) - Using Google's own deployment server.
# * [Deploy trained Keras or TensorFlow models using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/deploy-trained-keras-or-tensorflow-models-using-amazon-sagemaker/) - Using AWS to deploy TensorFlow ProtoBuffer models.
# * [Google ProtoBuf](https://developers.google.com/protocol-buffers/) - The file format used to store neural networks for deployment.
# 
# # Part 13.2.1: Train Model (optionally, outside of AWS)
# 
# A portion of this part will need to be run from [AWS SageMaker](https://aws.amazon.com/sagemaker/). To do this you will need to upload this IPYNB (for Module 13.2) to AWS Sage Maker and open it from Jupyter.  This complete process is demonstrated in the above YouTube video.
# 
# We begin by training a MPG dataset.  

# In[ ]:


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


# Next, we evaluate the RMSE.  The goal is more to show how to create a cloud API than to achieve a really low RMSE.

# In[ ]:


pred = model.predict(x_test)
# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(f"RMSE Score: {score}")


# Next we save the weights and structure of the neural network, as was demonstrated earlier in this course.  These two files are used to generate a ProtoBuf file that is used for the actual deployment.  We store it to two separate files because we ONLY want the structure and weights.  A single MD5 file, such as model.save(...) also contains training paramaters and other features that may cause version issues when uploading to AWS.  Remember, AWS may have a different version for TensorFlow than you do locally.  Usually AWS will have an older version.

# In[ ]:


save_path = "./dnn/"

model.save_weights(os.path.join(save_path,"mpg_model-weights.h5"))

# save neural network structure to JSON (no weights)
model_json = model.to_json()
with open(os.path.join(save_path,"mpg_model.json"), "w") as json_file:
    json_file.write(model_json)


# We will upload the two files generated to the **./dnn/** folder to AWS.  If you running the entire process from  AWS, then they will not need to be uploaded.
# 
# We also print out the values to one car, we will copy these later when we test the API.

# In[ ]:


x[0].tolist()


# # Part 13.2.2: Convert Model (must use AWS SageMaker Notebook)
# 
# To complete this portion you will need to be running from s Jupyter notebook on AWS SageMaker.  The following is based on an example from AWS documentation, but customized to this class example.

# ### Step 1. Set up
# 
# In the AWS Management Console, go to the Amazon SageMaker console. Choose Notebook Instances, and create a new notebook instance. Upload this notebook and set the kernel to conda_tensorflow_p36.
# 
# The get_execution_role function retrieves the AWS Identity and Access Management (IAM) role you created at the time of creating your notebook instance.

# In[ ]:


import boto3, re
from sagemaker import get_execution_role

role = get_execution_role()


# ### Step 2. Load the Keras model using the json and weights file
# 
# The following cell loads the necessary imports from AWS.  Note that we using "import keras" compared to the "import keras.tensorflow" advised for the rest of the course.  This is advised by AWS currently.

# In[ ]:


import keras
from keras.models import model_from_json


# Create a directory called keras_model, navigate to keras_model from the Jupyter notebook home, and upload the model.json and model-weights.h5 files (using the "Upload" menu on the Jupyter notebook home).

# In[ ]:


# get_ipython().system('mkdir keras_model')


# Navigate to keras_model from the Jupyter notebook home, and upload your model.json and model-weights.h5 files (using the "Upload" menu on the Jupyter notebook home). Use the files that you generated in step 2.

# In[ ]:


# get_ipython().system('ls keras_model')


# Make sure you've uploaded your model to the directory by this point.  If you saw no files at the above step, upload your files and rerun.

# In[ ]:


import tensorflow as tf

json_file = open('/home/ec2-user/SageMaker/keras_model/'+'mpg_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json,custom_objects={"GlorotUniform": tf.keras.initializers.glorot_uniform})


# In[ ]:


loaded_model.load_weights('/home/ec2-user/SageMaker/keras_model/mpg_model-weights.h5')
print("Loaded model from disk")


# ### Step 3. Export the Keras model to the TensorFlow ProtoBuf format (must use AWS SageMaker Notebook)
# 
# As you are probably noticing there are many ways to save a Keras neural network.  So far we've seen:
# 
# * YAML File - Structure only
# * JSON File - Structure only
# * H5 Complete Model
# * H5 Weights only
# 
# There is actually a fifth, which is the ProtoBuf format.  ProtoBuf is typically only used for deployment.  We will now convert the model we just loaded into this format.  

# In[ ]:


from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

# Note: This directory structure will need to be followed - see notes for the next section
model_version = '1'
export_dir = 'export/Servo/' + model_version


# It is very important that this export directory be empty.  Be careful, the following command deletes the entire expor directory. (this should be fine)

# In[ ]:


import shutil
shutil.rmtree(export_dir)


# In[ ]:


# Build the Protocol Buffer SavedModel at 'export_dir'
build = builder.SavedModelBuilder(export_dir)


# In[ ]:


# Create prediction signature to be used by TensorFlow Serving Predict API
signature = predict_signature_def(
    inputs={"inputs": loaded_model.input}, outputs={"score": loaded_model.output})


# In[ ]:


from keras import backend as K

with K.get_session() as sess:
    # Save the meta graph and variables
    build.add_meta_graph_and_variables(
        sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
    build.save()


# You will notive the **signature_def_map** this bridges any incompatabilities between the version of TensorFlow you were running locally and the AWS version.  You might need to add additional entries here.

# ### Step 4. Convert TensorFlow model to a SageMaker readable format (must use AWS SageMaker Notebook)
# 
# Move the TensorFlow exported model into a directory export\Servo. SageMaker will recognize this as a loadable TensorFlow model. Your directory and file structure should look like:

# In[ ]:


# get_ipython().system('ls export')


# In[ ]:


# get_ipython().system('ls export/Servo')


# In[ ]:


# get_ipython().system('ls export/Servo/1/variables')


# ####  Tar the entire directory and upload to S3

# In[ ]:


import tarfile
with tarfile.open('model.tar.gz', mode='w:gz') as archive:
    archive.add('export', recursive=True)


# Upload TAR file to S3.

# In[ ]:


import sagemaker

sagemaker_session = sagemaker.Session()
inputs = sagemaker_session.upload_data(path='model.tar.gz', key_prefix='model')


# ### Step 5. Deploy the trained model (must use AWS SageMaker Notebook)
# 
# The entry_point file "train.py" can be an empty Python file. The requirement will be removed at a later date.

# In[ ]:


# get_ipython().system('touch train.py')


# In[ ]:


from sagemaker.tensorflow.model import TensorFlowModel
sagemaker_model = TensorFlowModel(model_data = 's3://' + sagemaker_session.default_bucket() + '/model/model.tar.gz',
                                  role = role,
                                  framework_version = '1.12',
                                  entry_point = 'train.py')


# Note, the following command cake take 5-8 minutes to complete.

# In[ ]:


# get_ipython().run_cell_magic('time', '', "predictor = sagemaker_model.deploy(initial_instance_count=1,\n                                   instance_type='ml.m4.xlarge')")


# In[ ]:


predictor.endpoint


# Note: You will need to update the endpoint in the command below with the endpoint name from the output of the previous cell (e.g. sagemaker-tensorflow-2019-07-24-01-47-19-895)

# In[ ]:


endpoint_name = 'sagemaker-tensorflow-2019-08-05-03-29-25-591'


# In[ ]:


import sagemaker
from sagemaker.tensorflow.model import TensorFlowModel
predictor=sagemaker.tensorflow.model.TensorFlowPredictor(endpoint_name, sagemaker_session)


# # Part 13.2.3: Test Model Deployment (optionally, outside of AWS)

# In[1]:


import json
import boto3
import numpy as np
import io

endpoint_name = 'sagemaker-tensorflow-2019-08-05-03-29-25-591' # see above, must be set to current value

# Pick one of the following two cells to run based on how you will access...


# **Important, do not run both of the cells below!! Read comments**

# In[2]:


# If you access the API from outside of AWS SageMaker notebooks you must authenticate and specify region...
# (do not run both this cell and the next)

client = boto3.client('runtime.sagemaker', 
    region_name='us-east-1', # make sure to set correct region
    aws_access_key_id='AKIAYKSSG3L5P2H5EU77', # These you get from AWS, for your account
    aws_secret_access_key='1GYDRaE1o/nFfW2nF6jAJpWrd2R5Eut/d6fS6ruL')


# In[ ]:


# If you access from inside AWS in a notebook (do not run both this cell and the previous)
client = boto3.client('runtime.sagemaker', region_name='us-east-1') # make sure to set correct region


# ### Call the end point

# In[3]:


# Create a car based on one of the cars captured at beginning of this part.
data = [[8,307,130,3504,12,70,1]]

response = client.invoke_endpoint(EndpointName=endpoint_name, Body=json.dumps(data))
response_body = response['Body']
print(response_body.read())


# In[ ]:




