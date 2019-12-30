#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 2: Python for Machine Learning**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 2 Material
# 
# Main video lecture:
# 
# * Part 2.1: Introduction to Pandas [[Video]](https://www.youtube.com/watch?v=bN4UuCBdpZc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_1_python_pandas.ipynb)
# * Part 2.2: Categorical Values [[Video]](https://www.youtube.com/watch?v=4a1odDpG0Ho&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_2_pandas_cat.ipynb)
# * Part 2.3: Grouping, Sorting, and Shuffling in Python Pandas [[Video]](https://www.youtube.com/watch?v=YS4wm5gD8DM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_3_pandas_grouping.ipynb)
# * Part 2.4: Using Apply and Map in Pandas for Keras [[Video]](https://www.youtube.com/watch?v=XNCEZ4WaPBY&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_4_pandas_functional.ipynb)
# * **Part 2.5: Feature Engineering in Pandas for Deep Learning in Keras** [[Video]](https://www.youtube.com/watch?v=BWPTj4_Mi9E&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_02_5_pandas_features.ipynb)

# # Part 2.5: Feature Engineering

# Feature engineering is a very important part of machine learning.  Later in this course we will see some techniques for automatic feature engineering.  

# ## Calculated Fields
# 
# It is possible to add new fields to the dataframe that are calculated from the other fields.  We can create a new column that gives the weight in kilograms.  The equation to calculate a metric weight, given a weight in pounds is:
# 
# $ m_{(kg)} = m_{(lb)} \times 0.45359237 $
# 
# This can be used with the following Python code:

# In[1]:


import os
import pandas as pd

df = pd.read_csv(
    "https://data.heatonresearch.com/data/t81-558/auto-mpg.csv", 
    na_values=['NA', '?'])

df.insert(1, 'weight_kg', (df['weight'] * 0.45359237).astype(int))
df


# ## Google API Keys
# 
# Sometimes you will use external API's to obtain data.  The following examples show how to use the Google API keys to encode addresses for use with neural networks.  To use these, you will need your own Google API key.  The key I have below is not a real key, you need to put your own in there.  Google will ask for a credit card, but unless you use a very large number of lookups, there will be no actual cost.  YOU ARE NOT required to get an Google API key for this class, this only shows you how.  If you would like to get a Google API key, visit this site and obtain one for **geocode**.
# 
# [Google API Keys](https://developers.google.com/maps/documentation/embed/get-api-key)

# In[2]:


GOOGLE_KEY = 'Your Google API Key'


# # Other Examples: Dealing with Addresses
# 
# Addresses can be difficult to encode into a neural network.  There are many different approaches, and you must consider how you can transform the address into something more meaningful.  Map coordinates can be a good approach.  [Latitude and longitude](https://en.wikipedia.org/wiki/Geographic_coordinate_system) can be a useful encoding.  Thanks to the power of the Internet, it is relatively easy to transform an address into its latitude and longitude values.  The following code determines the coordinates of [Washington University](https://wustl.edu/):

# In[3]:


import requests

address = "1 Brookings Dr, St. Louis, MO 63130"

response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(GOOGLE_KEY,address))

resp_json_payload = response.json()

if 'error_message' in resp_json_payload:
    print(resp_json_payload['error_message'])
else:
    print(resp_json_payload['results'][0]['geometry']['location'])


# If latitude and longitude are simply fed into the neural network as two features, they might not be overly helpful.  These two values would allow your neural network to cluster locations on a map.  Sometimes cluster locations on a map can be useful.  Consider the percentage of the population that smokes in the USA by state:
# 
# ![Smokers by State](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_6_smokers.png "Smokers by State")
# 
# The above map shows that certain behaviors, like smoking, can be clustered by global region. 
# 
# However, often you will want to transform the coordinates into distances.  It is reasonably easy to estimate the distance between any two points on Earth by using the [great circle distance](https://en.wikipedia.org/wiki/Great-circle_distance) between any two points on a sphere:
# 
# The following code implements this formula:
# 
# $\Delta\sigma=\arccos\bigl(\sin\phi_1\cdot\sin\phi_2+\cos\phi_1\cdot\cos\phi_2\cdot\cos(\Delta\lambda)\bigr)$
# 
# $d = r \, \Delta\sigma$

# In[4]:


from math import sin, cos, sqrt, atan2, radians

# Distance function
def distance_lat_lng(lat1,lng1,lat2,lng2):
    # approximate radius of earth in km
    R = 6373.0

    # degrees to radians (lat/lon are in degrees)
    lat1 = radians(lat1)
    lng1 = radians(lng1)
    lat2 = radians(lat2)
    lng2 = radians(lng2)

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

# Find lat lon for address
def lookup_lat_lng(address):
    response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}'.format(GOOGLE_KEY,address))
    json = response.json()
    if len(json['results']) == 0:
        print("Can't find: {}".format(address))
        return 0,0
    map = json['results'][0]['geometry']['location']
    return map['lat'],map['lng']


# Distance between two locations

import requests

address1 = "1 Brookings Dr, St. Louis, MO 63130" 
address2 = "3301 College Ave, Fort Lauderdale, FL 33314"

lat1, lng1 = lookup_lat_lng(address1)
lat2, lng2 = lookup_lat_lng(address2)

print("Distance, St. Louis, MO to Ft. Lauderdale, FL: {} km".format(
        distance_lat_lng(lat1,lng1,lat2,lng2)))


# Distances can be useful to encode addresses as.  You must consider what distance might be useful for your dataset.  Consider:
# 
# * Distance to major metropolitan area
# * Distance to competitor
# * Distance to distribution center
# * Distance to retail outlet
# 
# The following code calculates the distance between 10 universities and washu:

# In[5]:


# Encoding other universities by their distance to Washington University

schools = [
    ["Princeton University, Princeton, NJ 08544", 'Princeton'],
    ["Massachusetts Hall, Cambridge, MA 02138", 'Harvard'],
    ["5801 S Ellis Ave, Chicago, IL 60637", 'University of Chicago'],
    ["Yale, New Haven, CT 06520", 'Yale'],
    ["116th St & Broadway, New York, NY 10027", 'Columbia University'],
    ["450 Serra Mall, Stanford, CA 94305", 'Stanford'],
    ["77 Massachusetts Ave, Cambridge, MA 02139", 'MIT'],
    ["Duke University, Durham, NC 27708", 'Duke University'],
    ["University of Pennsylvania, Philadelphia, PA 19104", 'University of Pennsylvania'],
    ["Johns Hopkins University, Baltimore, MD 21218", 'Johns Hopkins']
]

lat1, lng1 = lookup_lat_lng("1 Brookings Dr, St. Louis, MO 63130")

for address, name in schools:
    lat2,lng2 = lookup_lat_lng(address)
    dist = distance_lat_lng(lat1,lng1,lat2,lng2)
    print("School '{}', distance to wustl is: {}".format(name,dist))


# In[ ]:




