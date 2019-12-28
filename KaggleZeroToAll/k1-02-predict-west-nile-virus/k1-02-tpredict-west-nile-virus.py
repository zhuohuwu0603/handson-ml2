#!/usr/bin/env python
# coding: utf-8

# # West Nile Virus Prediction
# https://www.kaggle.com/c/predict-west-nile-virus
# 
# base code : https://www.kaggle.com/duttaroy/enhanced

# ## 1. Problem/data description
# 
# In this competition, you will be analyzing weather data and GIS data and predicting whether or not West Nile virus is present, for a given time, location, and species. 
# 
# Every year from late-May to early-October, public health workers in Chicago setup mosquito traps scattered across the city. Every week from Monday through Wednesday, these traps collect mosquitos, and the mosquitos are tested for the presence of West Nile virus before the end of the week. The test results include the number of mosquitos, the mosquitos species, and whether or not West Nile virus is present in the cohort. 
# 
# ## train.csv, test.csv 
# The training set consists of data from 2007, 2009, 2011, and 2013, while in the test set you are requested to predict the test results for 2008, 2010, 2012, and 2014.
# 
# * Id: the id of the record
# * Date: date that the WNV test is performed
# * Address: approximate address of the location of trap. This is used to send to the GeoCoder. 
# * Species: the species of mosquitos
# * Block: block number of address
# * Street: street name
# * Trap: Id of the trap
# * AddressNumberAndStreet: approximate address returned from GeoCoder
# * Latitude, Longitude: Latitude and Longitude returned from GeoCoder
# * AddressAccuracy: accuracy returned from GeoCoder
# * NumMosquitos: number of mosquitoes caught in this trap
# * WnvPresent: whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. 
# 
# ## spray.csv
# GIS data of spraying efforts in 2011 and 2013
# * Date, Time: the date and time of the spray
# * Latitude, Longitude: the Latitude and Longitude of the spray
# 
# ## weather.csv 
# weather data from 2007 to 2014. Column descriptions in noaa_weather_qclcd_documentation.pdf. 

# ## 2. Loading data
# Show how to download and load them in Kerans, TensorFlow, etc.

# In[1]:


import os
import csv
import math
import pickle
import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')


# ### data convert function and fill missing value

# In[2]:


def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()

def ll(text):
     return int(float(text)*100)/100

def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "-":
        return None
    if text == "T":
        return TRACE
    return float(text)

def impute_missing_weather_station_values(weather):
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]


# ### load function
# 
# ```python
# for line in csv.DictReader(open("dataPath")): # open csvfile and access line by line
#     for name, converter in feature_dict: # feature_dict contain col name and data converter to use
#         line[name] = converter(line[name]) # convert data
#    
# ```

# In[3]:


def load_weather():
    weather = {}
    for line in csv.DictReader(open("input/weather.csv")):
        for name, converter in {"Date" : date,
                                "Tmax" : float,"Tmin" : float,"Tavg" : float,
                                "DewPoint" : float, "WetBulb" : float,
                                "PrecipTotal" : precip,"Sunrise" : precip,"Sunset" : precip,
                                "Depart" : float, "Heat" : precip,"Cool" : precip,
                                "ResultSpeed" : float,"ResultDir" : float,"AvgSpeed" : float,
                                "StnPressure" : float, "SeaLevel" : float}.items():
            
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
            
        station = int(line["Station"]) - 1
        
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
            
        weather[dt][station] = line
    impute_missing_weather_station_values(weather) # fill missing values       
    return weather
    
def load_train():
    training = []
    for line in csv.DictReader(open("input/train.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : ll, "Longitude" : ll,
                                "NumMosquitos" : int, "WnvPresent" : int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training
    
def load_test():
    training = []
    for line in csv.DictReader(open("input/test.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : ll, "Longitude" : ll}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training


# In[4]:


def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std
    
def scaled_count(record):
    SCALE = 9.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))

def get_closest_station(lat, long):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, long])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1)
    return np.argmin(dist2)


# In[5]:


species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000", 
              'CULEX PIPIENS'   : "001000", 
              'CULEX PIPIENS/RESTUANS' : "101000", 
              'CULEX ERRATICUS' : "000100", 
              'CULEX SALINARIUS': "000010", 
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001100"} # hack! https://www.kaggle.com/c/predict-west-nile-virus/discussion/13810

# species name: vector
    
        
def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        date2 = np.sin((2 * math.pi * date.day) / 365 * 24)
        date3 = np.cos((2 * math.pi * date.day) / 365 * 24)
        date4 = np.sin((2 * math.pi * date.month) / 365)
        
        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date4, date.day, date.weekday(), date2, date3, lat, longi]
        
        # Look at a selection of past weather values
        for days_ago in [1,2,3,5,8,12]:
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax", "Tmin", "Tavg", "DewPoint", "WetBulb",
                        "PrecipTotal", "Depart", "Sunrise", "Sunset",
                        "Heat", "Cool", "ResultSpeed", "ResultDir"]:
                
                station = get_closest_station(lat, longi)
                case.append(weather[day][station][obs])
                
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        
        '''
        Weight each observation by the number of mosquitos seen.
        Test data Doesn't have this column, so in that case use 1. 
        This accidentally Takes into account multiple entries that 
        result from >50 mosquitos on one day.
        '''
        
        for repeat in range(scaled_count(b)):
            X.append(case)    
    X = np.asarray(X, dtype=np.float32)
    return X
    
def assemble_y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)    
    return np.asarray(y, dtype=np.float32).reshape(-1,1)


# In[6]:


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


# In[7]:


# use only once for data dump

train_data = load_train() 
test_data = load_test()
weather_data = load_weather()

x_train = assemble_X(train_data, weather_data)
x_test = assemble_X(test_data, weather_data) 

y_train = assemble_y(train_data)

mean, std = normalize(x_train)
mean, std = normalize(x_test, mean, std)

cache_data(test_data, './test_data.dump')

cache_data(x_train, './x_train.dump')
cache_data(x_test, './x_test.dump')
cache_data(y_train, './y_train.dump')


# In[8]:


test_data = restore_data('./test_data.dump') # for submission

x_train = restore_data('./x_train.dump')
x_test = restore_data('./x_test.dump')
y_train = restore_data('./y_train.dump')


# ## 3. Model
# Model implementation. It can be divided to several small sections.

# In[9]:


input_size = x_train.shape[1]
print(x_train.shape)


# In[10]:


import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = tf.placeholder(tf.float32, shape=[None, input_size], name="X")
Y = tf.placeholder(tf.float32, shape=[None, 1], name="Y")
dropout = tf.placeholder(tf.float32, shape=None)

W1 = tf.Variable(tf.random_normal([input_size, 100], name='w1'))
W2 = tf.Variable(tf.random_normal([100, 100], name='w2'))
W3 = tf.Variable(tf.random_normal([100, 1], name='w3'))

b1 = tf.Variable(tf.random_normal([100], name='b1'))
b2 = tf.Variable(tf.random_normal([100], name='b2'))
b3 = tf.Variable(tf.random_normal([1], name='b3'))


layer1 = tf.matmul(X, W1) + b1
layer1 = tf.nn.dropout(layer1, dropout)
layer2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.dropout(layer2, dropout)
layer3 = tf.matmul(layer2, W3) + b3

pred = tf.nn.sigmoid(layer3)
                 
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer3, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


# In[11]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 501
batch_size = 500

for e in range(epochs):
    avg_cost = 0 

    for step in range(0, len(x_train), batch_size):
        batch_mask = np.random.choice(len(x_train), batch_size) # For dataset shuffle
        
        feed_dict = {X: x_train[batch_mask], 
                     Y: y_train[batch_mask], dropout: 0.3}
        
        _, c = sess.run([optimizer, cost], feed_dict=feed_dict)
        
        avg_cost += c
        
    
    if e % 100 == 0:
        avg_cost = avg_cost / (len(x_train) / batch_size) 
        print("epoch: {} cost: {:.5f}".format(e, avg_cost))


# ## 4. Results
# show the result

# In[12]:


predictions = sess.run(pred, feed_dict={X: x_test, dropout: 1})[:, 0]  


# In[13]:


with open("west_nile.csv", "w") as csv_file:
    out = csv.writer(csv_file)
    out.writerow(["Id","WnvPresent"])
    for row, p in zip(test_data, predictions):
        out.writerow([row["Id"], p])


# ## 5. Future work/exercises

# Ensemble, Parameter Tuning
