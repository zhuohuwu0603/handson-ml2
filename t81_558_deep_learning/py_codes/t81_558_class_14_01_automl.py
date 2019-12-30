#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 14: Other Neural Network Techniques**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 14 Video Material
# 
# * **Part 14.1: What is AutoML** [[Video]](https://www.youtube.com/watch?v=TFUysIR5AB0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_01_automl.ipynb)
# * Part 14.2: Using Denoising AutoEncoders in Keras [[Video]](https://www.youtube.com/watch?v=4bTSu6_fucc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_02_auto_encode.ipynb)
# * Part 14.3: Training an Intrusion Detection System with KDD99 [[Video]](https://www.youtube.com/watch?v=1ySn6h2A68I&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_03_anomaly.ipynb)
# * Part 14.4: Anomaly Detection in Keras [[Video]](https://www.youtube.com/watch?v=VgyKQ5MTDFc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_04_ids_kdd99.ipynb)
# * Part 14.5: The Deep Learning Technologies I am Excited About [[Video]]() [[Notebook]](t81_558_class_14_05_new_tech.ipynb)
# 
# 

# # Part 14.1: What is AutoML
# 
# Automatic Machine Learning (AutoML) attempts to use machine learning to automate itself.  Data is passed to the AutoML application in raw form and models are automatically generated.
# 
# ### AutoML from your Local Computer
# 
# The following AutoML applications are commercial.
# 
# * [Rapid Miner](https://rapidminer.com/educational-program/) - Free student version available.
# * [Dataiku](https://www.dataiku.com/dss/editions/) - Free community version available.
# * [DataRobot](https://www.datarobot.com/) - Commercial
# * [H2O Driverless](https://www.h2o.ai/products/h2o-driverless-ai/) - Commercial
# 
# ### AutoML from Google Cloud
# 
# * [Google Cloud AutoML Tutorial](https://cloud.google.com/vision/automl/docs/tutorial)
# 
# 

# ### A Simple AutoML System
# 
# The following program is a very simple implementation of AutoML.  It is able to take RAW tabular data and construct a neural network.  
# 
# We begin by defining a class that abstracts the differences between reading CSV over local file system or HTTP/HTTPS.

# In[1]:


import requests
import csv

class CSVSource():
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        if self.filename.lower().startswith("https:") or self.filename.lower().startswith("https:"):
            r = requests.get(self.filename, stream=True)
            self.infile = (line.decode('utf-8') for line in r.iter_lines())
            return csv.reader(self.infile)
        else:
            self.infile = codecs.open(self.filename, "r", "utf-8")
            return csv.reader(self.infile)
    def __exit__(self, type, value, traceback):
        self.infile.close()
        


# The following code analyzes the tabular data and determines a way of encoding the feature vector.

# In[2]:


import csv
import codecs
import math
import os
import re
from numpy import genfromtxt

MAX_UNIQUES = 200

INPUT_ENCODING = 'latin-1'

CMD_CAT_DUMMY = 'dummy-cat'
CMD_CAT_NUMERIC = 'numeric-cat'
CMD_IGNORE = 'ignore'
CMD_MAP = 'map'
CMD_PASS = 'pass'
CMD_BITS = 'bits'

CONTROL_INDEX = 'index'
CONTROL_NAME = 'name'
CONTROL_COMMAND = 'command'
CONTROL_TYPE = 'type'
CONTROL_LENGTH = 'length'
CONTROL_UNIQUE_COUNT = 'unique_count'
CONTROL_UNIQUE_LIST = 'unique_list'
CONTROL_MISSING = 'missing'
CONTROL_MEAN = 'mean'
CONTROL_SDEV = 'sdev'


MAP_SKIP = True
MISSING_SKIP = False

current_row = 0

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def isna(s):
    return s.upper() == 'NA' or s.upper() == 'N/A' or s.upper() == 'NULL' or len(s) < 1 or s.upper() == '?'

def analyze(filename):
    fields = []
    first_header = None

    # Pass 1 (very short. First, look at the first row of each of the provided files.
    # Build field blocks from the first file, and ensure that other files
    # match the first one.
    
    with CSVSource(filename) as reader:
        header = next(reader)

        if first_header is None:
            first_header = header

            for idx, field_name in enumerate(header):
                fields.append({
                    'name': field_name,
                    'command': '?',
                    'index': idx,
                    'type': None,
                    'missing': False,
                    'unique': {},
                    'count': 0,
                    'mean': '',
                    'sum': 0,
                    'sdev': '',
                    'length': 0})
        else:
            for x, y in zip(header, first_header):
                if x != y:
                    raise ValueError('The headers do not match on the input files')


    # Pass 2 over the files
    with CSVSource(filename) as reader:
        next(reader)

        # Determine types and calculate sum
        for row in reader:
            if len(row) != len(fields):
                continue
            for data, field_info in zip(row, fields):
                data = data.strip()
                field_info['length'] = max(len(data),field_info['length'])
                if len(data) < 1 or data.upper() == 'NULL' or isna(data):
                    field_info[CONTROL_MISSING] = True
                else:
                    if not is_number(data):
                        field_info['type'] = 'text'

                    # Track the unique values and counts per unique item
                    cat_map = field_info['unique']
                    if data in cat_map:
                        cat_map[data]['count']+=1
                    else:
                        cat_map[data] = {'name':data,'count':1}

                    if field_info['type'] != 'text':
                        field_info['count'] += 1
                        field_info['sum'] += float(data)

    # Finalize types
    for field in fields:
        if field['type'] is None:
            field['type'] = 'numeric'
        field[CONTROL_UNIQUE_COUNT] = len(field['unique'])

    # Calculate mean
    for field in fields:
        if field['type'] == 'numeric' and field['count'] > 0:
            field['mean'] = field['sum'] / field['count']


    # Pass 3 over the files, calculate standard deviation and finailize fields.
    sums = [0] * len(fields)
    
    with CSVSource(filename) as reader:
        next(reader)

        for row in reader:
            if len(row) != len(fields):
                continue
            for data, field_info in zip(row, fields):
                data = data.strip()
                if field_info['type'] == 'numeric' and len(data) > 0 and not isna(data):
                    sums[field_info['index']] += (float(data) - field_info['mean']) ** 2

    # Examine fields
    for idx, field in enumerate(fields):
        if field['type'] == 'numeric' and field['count'] > 0:
            field['sdev'] = math.sqrt(sums[field['index']] / field['count'])

        # Assign a default command
        if field['name'] == 'ID' or field['name'] == 'FOLD':
            field['command'] = 'pass'
        elif "DATE" in field['name'].upper():
            field['command'] = 'date'
        elif field['unique_count'] == 2 and field['type'] == 'numeric':
            field['command'] = CMD_PASS
        elif field['type'] == 'numeric' and field['unique_count'] < 25:
            field['command'] = CMD_CAT_DUMMY
        elif field['type'] == 'numeric':
            field['command'] = 'zscore'
        elif field['type'] == 'text' and field['unique_count'] <= MAX_UNIQUES:
            field['command'] = CMD_CAT_DUMMY
        else:
            field['command'] = CMD_IGNORE

    return fields

def write_control_file(filename, fields):
    with codecs.open(filename, "w", "utf-8") as outfile:
        writer = csv.writer(outfile,quoting=csv.QUOTE_NONNUMERIC)

        writer.writerow([CONTROL_INDEX, CONTROL_NAME, CONTROL_COMMAND, CONTROL_TYPE, CONTROL_LENGTH, CONTROL_UNIQUE_COUNT, CONTROL_MISSING, CONTROL_MEAN, CONTROL_SDEV])
        for field in fields:

            # Write the main row for the field (left-justified)
            writer.writerow([field[CONTROL_INDEX], field[CONTROL_NAME], field[CONTROL_COMMAND], field[CONTROL_TYPE], field[CONTROL_LENGTH], field[CONTROL_UNIQUE_COUNT],
                             field[CONTROL_MISSING], field[CONTROL_MEAN], field[CONTROL_SDEV]])

            # Write out any needed category information
            if field[CONTROL_UNIQUE_COUNT] <= MAX_UNIQUES:
                sorted_cat = field['unique'].values()
                sorted_cat = sorted(sorted_cat, key=lambda k: k[CONTROL_NAME])
                for category in sorted_cat:
                    writer.writerow(["","",category[CONTROL_NAME],category['count']])
            else:
                catagories = ""



def read_control_file(filename):
    with codecs.open(filename, "r", "utf-8") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        lookup = {}
        for i, name in enumerate(header):
            lookup[name] = i

        fields = []
        categories = {}

        for row in reader:
            if row[0] == '':
                name = row[2]
                mp = '' if len(row)<=4 else row[4]
                categories[name] = {'name':name,'count':int(row[3]),'map':mp}
                if len(categories)>0:
                    field[CONTROL_UNIQUE_LIST] = sorted(categories.keys())
            else:
                # New field
                field = {}
                categories = {}
                field['unique'] = categories
                for key in lookup.keys():
                    value = row[lookup[key]]
                    if key in ['unique_count', 'count', 'index', 'length']:
                        value = int(value)
                    elif key in ['sdev', 'mean', 'sum']:
                        if len(value) > 0:
                            value = float(value)
                    field[key] = value

                field['len'] = -1
                fields.append(field)
        return fields

def header_cat_dummy(field, header):
    name = str(field['name'])

    for c in field['unique']:
        dname = "{}-D:{}".format(name, c)
        header.append(dname)

def header_bits(field, header):
    for i in range(field['length']):
        header.append("{}-B:{}".format(field['name'], i))


def header_other(field, header):
    header.append(field['name'])


def column_zscore(field,write_row,value,has_na):
    if isna(value) or field['sdev'] == 0:
        #write_row.append('NA')
        #has_na = True
        write_row.append(0)
    elif not is_number(value):
        raise ValueError("Row {}: Non-numeric for zscore: {} on field {}".format(current_row,value,field['name']))
    else:
        value = (float(value) - field['mean']) / field['sdev']
        write_row.append(value)
    return has_na

def column_cat_numeric(field,write_row,value,has_na):
    if CONTROL_UNIQUE_LIST not in field:
        raise ValueError("No value list, can't encode {} to numeric categorical.".format(field[CONTROL_NAME]))

    if value not in field[CONTROL_UNIQUE_LIST]:
        write_row.append("NA")
        has_na = True
    else:
        idx = field[CONTROL_UNIQUE_LIST].index(value)
        write_row.append('class-' + str(idx))
    return has_na

def column_map(field,write_row,value,has_na):
    if value in field['unique']:
        mapping = field['unique'][value]['map']
        write_row.append(mapping)
    else:
        write_row.append("NA")
        return True
    return has_na


def column_cat_dummy(field,write_row,value,has_na):
    for c in field['unique']:
        write_row.append(0 if value != c else 1)
    return has_na

def column_bits(field,write_row,value,has_na):
    if len(value)!=field['length']:
        raise ValueError("Invalid bits length: {}, expected: {}".format(
            len(value),field['length']))

    for c in value:
        if c == 'Y':
            write_row.append(1)
        elif c == 'N':
            write_row.append(-1)
        else:
            write_row.append(0)
    return has_na

def transform_file(input_file, output_file, fields):
    print("**Transforming to file: {}".format(output_file))
    with CSVSource(input_file) as reader,             codecs.open(output_file, "w", "utf-8") as outfile:
        writer = csv.writer(outfile)

        next(reader)
        header = []

        # Write the header
        for field in fields:
            if field['command'] == CMD_IGNORE:
                pass
            elif field['command'] == CMD_CAT_DUMMY:
                header_cat_dummy(field,header)
            elif field['command'] == CMD_BITS:
                header_bits(field,header)
            else:
                header_other(field,header)

        print("Columns generated: {}".format(len(header)))

        writer.writerow(header)
        line_count = 0
        lines_skipped = 0

        # Process the actual file
        current_row = -1
        header_len = len(header)
        for row in reader:
            if len(row) != len(fields):
                continue
                
            current_row+=1
            has_na = False
            write_row = []
            for field in fields:
                value = row[field['index']].strip()

                cmd = field['command']
                if cmd == 'zscore':
                    has_na = column_zscore(field,write_row,value, has_na)
                elif cmd == CMD_CAT_NUMERIC:
                    has_na = column_cat_numeric(field,write_row,value, has_na)
                elif cmd == CMD_IGNORE:
                    pass
                elif cmd == CMD_MAP:
                    has_na = column_map(field,write_row,value, has_na)
                elif cmd == CMD_PASS:
                    write_row.append(value)
                elif cmd == 'date':
                    write_row.append(str(value[-4:]))
                elif cmd == CMD_CAT_DUMMY:
                    has_na = column_cat_dummy(field,write_row,value, has_na)
                elif cmd == CMD_BITS:
                    has_na = column_bits(field,write_row,value,has_na)
                else:
                    raise ValueError("Unknown command: {}, stopping.".format(cmd))


            if MISSING_SKIP and has_na:
                lines_skipped += 1
                pass
            else:
                line_count += 1
                writer.writerow(write_row)

                # Double check!
                if len(write_row) != header_len:
                    raise ValueError("Inconsistant column count near line: {}, only had: {}".format(line_count,len(write_row)))

    print("Data rows written: {}, skipped: {}".format(line_count,lines_skipped))
    print()

def find_field(control, name):
    for field in control:
        if field['name'] == name:
            return field
    return None

def find_transformed_fields(header, name):
    y = []
    x = []
    for idx, field in enumerate(header):
        if field.startswith(name + '-') or field==name:
            y.append(idx)
        else:
            x.append(idx)
            
    return x,y

def process_for_fit(control, transformed_file, target):
    
    with CSVSource(transformed_file) as reader:
        header = next(reader)
    
    field = find_field(control, target)
    if field is None:
        raise ValueError(f"Unknown target column specified:{target}")

    if field['command'] == 'dummy-cat':
        print(f"Performing classification on: {target}")
    else:
        print(f"Performing regression on: {target}")
        
    x_ids, y_ids = find_transformed_fields(header, target)
    
    x = genfromtxt("transformed.csv", delimiter=',', skip_header=1)
    y = x[:,y_ids]
    x = x[:,x_ids]
    return x,y


# The following code takes the data processed from above and trains a neural network.

# In[3]:


import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn import metrics
from sklearn.model_selection import KFold

def generate_network(x,y,task):
    model = Sequential()
    model.add(Dense(50, input_dim=x.shape[1], activation='relu')) # Hidden 1
    model.add(Dense(25, activation='relu')) # Hidden 2
    
    if task == 'classify':
        model.add(Dense(y.shape[1],activation='softmax')) # Output
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    else:
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
    return model

def cross_validate(x,y,folds,task):
    
    if task == 'classify':
        cats = y.argmax(axis=1)
        kf = StratifiedKFold(folds, shuffle=True, random_state=42).split(x,cats)
    else:
        kf = KFold(folds, shuffle=True, random_state=42).split(x) 
    
    oos_y = []
    oos_pred = []
    fold = 0
    
    for train, test in kf:
        fold+=1
        print(f"Fold #{fold}")

        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]

        model = generate_network(x,y,task)
        model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=0,epochs=500)

        pred = model.predict(x_test)

        oos_y.append(y_test)
        
        if task == 'classify':
            pred = np.argmax(pred,axis=1) # raw probabilities to chosen class (highest probability)
        oos_pred.append(pred)  

        if task == 'classify':
            # Measure this fold's accuracy
            y_compare = np.argmax(y_test,axis=1) # For accuracy calculation
            score = metrics.accuracy_score(y_compare, pred)
            print(f"Fold score (accuracy): {score}")
        else:
            score = np.sqrt(metrics.mean_squared_error(pred,y_test))
            print(f"Fold score (RMSE): {score}")
            
        
    # Build the oos prediction list and calculate the error.
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    
    if task == 'classify':
        oos_y_compare = np.argmax(oos_y,axis=1) # For accuracy calculation
        score = metrics.accuracy_score(oos_y_compare, oos_pred)
        print(f"Final score (accuracy): {score}") 
    else:
        score = np.sqrt(metrics.mean_squared_error(oos_y, oos_pred))
        print(f"Final score (RMSE): {score}")


# ### Running My Sample AutoML Program
# 
# These three variables are all you really need to define.

# In[4]:


SOURCE_DATA = 'https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv'
TARGET_FIELD = 'product'
TASK = 'classify'

#SOURCE_DATA = 'https://data.heatonresearch.com/data/t81-558/iris.csv'
#TARGET_FIELD = 'species'
#TASK = 'classify'

#SOURCE_DATA = 'https://data.heatonresearch.com/data/t81-558/auto-mpg.csv'
#TARGET_FIELD = 'mpg'
#TASK = 'reg'


# The following lines of code analyze your source data file and figure out how to encode each column.  The result is a control file that you can modify to control how each column is handled.  The below code should only be run ONCE to generate a control file as a starting point for you to modify.

# In[5]:


import csv
import requests
import codecs

control = analyze(SOURCE_DATA)
write_control_file("control.csv",control)


# If your control file is already create, you can start here (after defining the above constants).  Do not rerun the previous section, as it will overwrite your control file.  Now transform the data.

# In[6]:


control = read_control_file("control.csv")
transform_file(SOURCE_DATA,"transformed.csv",control)


# Load the transformed data into properly preprocessed $x$ and $y$. 

# In[7]:


x,y = process_for_fit(control, "transformed.csv", TARGET_FIELD)
print(x.shape)
print(y.shape)


# Double check to be sure there are no missing values remaining.

# In[8]:


import numpy as np
np.isnan(x).any()


# We are now ready to cross validate and train.

# In[9]:


cross_validate(x,y,5,TASK)    


# In[ ]:




