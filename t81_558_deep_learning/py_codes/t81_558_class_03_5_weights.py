#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 3: Introduction to TensorFlow**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 3 Material
# 
# * Part 3.1: Deep Learning and Neural Network Introduction [[Video]](https://www.youtube.com/watch?v=zYnI4iWRmpc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_1_neural_net.ipynb)
# * Part 3.2: Introduction to Tensorflow and Keras [[Video]](https://www.youtube.com/watch?v=PsE73jk55cE&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_2_keras.ipynb)
# * Part 3.3: Saving and Loading a Keras Neural Network [[Video]](https://www.youtube.com/watch?v=-9QfbGM1qGw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_3_save_load.ipynb)
# * Part 3.4: Early Stopping in Keras to Prevent Overfitting [[Video]](https://www.youtube.com/watch?v=m1LNunuI2fk&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_4_early_stop.ipynb)
# * **Part 3.5: Extracting Weights and Manual Calculation** [[Video]](https://www.youtube.com/watch?v=7PWgx16kH8s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_5_weights.ipynb)

# # Part 3.5: Extracting Keras Weights and Manual Neural Network Calculation
# 
# In this section we will build a neural network and analyze it down the individual weights.  We will train a simple neural network that learns the XOR function.  It is not hard to simply hand-code the neurons to provide an [XOR function](https://en.wikipedia.org/wiki/Exclusive_or); however, for simplicity, we will allow Keras to train this network for us.  We will just use 100K epochs on the ADAM optimizer.  This is massive overkill, but it gets the result, and our focus here is not on tuning.  The neural network is small.  Two inputs, two hidden neurons, and a single output.

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

# Create a dataset for the XOR function
x = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])

y = np.array([
    0,
    1,
    1,
    0
])

# Build the network
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

done = False
cycle = 1

while not done:
    print("Cycle #{}".format(cycle))
    cycle+=1
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu')) 
    model.add(Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x,y,verbose=0,epochs=10000)

    # Predict
    pred = model.predict(x)
    
    # Check if successful.  It takes several runs with this small of a network
    done = pred[0]<0.01 and pred[3]<0.01 and pred[1] > 0.9 and pred[2] > 0.9 
    print(pred)


# In[2]:


pred[3]


# The output above should have two numbers near 0.0 for the first and forth spots (input [[0,0]] and [[1,1]]).  The middle two numbers should be near 1.0 (input [[1,0]] and [[0,1]]).  These numbers are in scientific notation.  Due to random starting weights, it is sometimes necessary to run the above through several cycles to get a good result.
# 
# Now that the neural network is trained, lets dump the weights.  

# In[3]:


# Dump weights
for layerNum, layer in enumerate(model.layers):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    
    for toNeuronNum, bias in enumerate(biases):
        print(f'{layerNum}B -> L{layerNum+1}N{toNeuronNum}: {bias}')
    
    for fromNeuronNum, wgt in enumerate(weights):
        for toNeuronNum, wgt2 in enumerate(wgt):
            print(f'L{layerNum}N{fromNeuronNum} -> L{layerNum+1}N{toNeuronNum} = {wgt2}')


# If you rerun this, you probably get different weights.  There are many ways to solve the XOR function.
# 
# In the next section, we copy/paste the weights from above and recreate the calculations done by the neural network.  Because weights can change with each training, the weights used for the below code came from this:
# 
# ```
# 0B -> L1N0: -1.2913415431976318
# 0B -> L1N1: -3.021530048386012e-08
# L0N0 -> L1N0 = 1.2913416624069214
# L0N0 -> L1N1 = 1.1912699937820435
# L0N1 -> L1N0 = 1.2913411855697632
# L0N1 -> L1N1 = 1.1912697553634644
# 1B -> L2N0: 7.626241297587034e-36
# L1N0 -> L2N0 = -1.548777461051941
# L1N1 -> L2N0 = 0.8394404649734497
# ```

# In[4]:


input0 = 0
input1 = 1

hidden0Sum = (input0*1.3)+(input1*1.3)+(-1.3)
hidden1Sum = (input0*1.2)+(input1*1.2)+(0)

print(hidden0Sum) # 0
print(hidden1Sum) # 1.2

hidden0 = max(0,hidden0Sum)
hidden1 = max(0,hidden1Sum)

print(hidden0) # 0
print(hidden1) # 1.2

outputSum = (hidden0*-1.6)+(hidden1*0.8)+(0)
print(outputSum) # 0.96

output = max(0,outputSum)

print(output) # 0.96


# In[ ]:




