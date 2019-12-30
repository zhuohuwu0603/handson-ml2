#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 3: Introduction to TensorFlow**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 3 Material
# 
# * **Part 3.1: Deep Learning and Neural Network Introduction** [[Video]](https://www.youtube.com/watch?v=zYnI4iWRmpc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_1_neural_net.ipynb)
# * Part 3.2: Introduction to Tensorflow and Keras [[Video]](https://www.youtube.com/watch?v=PsE73jk55cE&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_2_keras.ipynb)
# * Part 3.3: Saving and Loading a Keras Neural Network [[Video]](https://www.youtube.com/watch?v=-9QfbGM1qGw&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_3_save_load.ipynb)
# * Part 3.4: Early Stopping in Keras to Prevent Overfitting [[Video]](https://www.youtube.com/watch?v=m1LNunuI2fk&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_4_early_stop.ipynb)
# * Part 3.5: Extracting Weights and Manual Calculation [[Video]](https://www.youtube.com/watch?v=7PWgx16kH8s&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_03_5_weights.ipynb)

# # Part 3.1: Deep Learning and Neural Network Introduction
# 
# Neural networks were one of the first machine learning models.  Their popularity has fallen twice and is now on its third rise.  Deep learning implies the use of neural networks.  The "deep" in deep learning refers to a neural network with many hidden layers.  Because neural networks have been around for so long, they have quite a bit of baggage.  Many different training algorithms, activation/transfer functions, and structures have been added over the years.  This course is only concerned with the latest, most current state of the art techniques for deep neural networks.  I am not going to spend any time discussing the history of neural networks.  If you would like to learn about some of the more classic structures of neural networks, there are several chapters dedicated to this in your course book.  For the latest technology, I wrote an article for the Society of Actuaries on deep learning as the [third generation of neural networks](https://www.soa.org/Library/Newsletters/Predictive-Analytics-and-Futurism/2015/december/paf-iss12.pdf).
# 
# Neural networks accept input and produce output.  The input to a neural network is called the feature vector.  The size of this vector is always a fixed length.  Changing the size of the feature vector means recreating the entire neural network.  Though the feature vector is called a "vector," this is not always the case.  A vector implies a 1D array.  Historically the input to a neural network was always 1D.  However, with modern neural networks you might see inputs, such as:
# 
# * **1D Vector** - Classic input to a neural network, similar to rows in a spreadsheet.  Common in predictive modeling.
# * **2D Matrix** - Grayscale image input to a convolutional neural network (CNN).
# * **3D Matrix** - Color image input to a convolutional neural network (CNN).
# * **nD Matrix** - Higher order input to a CNN.
# 
# Prior to CNN's, the image input was sent to a neural network simply by squashing the image matrix into a long array by placing the image's rows side-by-side.  CNNs are different, as the nD matrix literally passes through the neural network layers.
# 
# Initially this course will focus upon 1D input to neural networks.  However, later sessions will focus more heavily upon higher dimension input.
# 
# **Dimensions** The term dimension can be confusing in neural networks.  In the sense of a 1D input vector, dimension refers to how many elements are in that 1D array.  For example, a neural network with 10 input neurons has 10 dimensions.  However, now that we have CNN's, the input has dimensions too.  The input to the neural network will *usually* have 1, 2 or 3 dimensions.  4 or more dimensions is unusual.  You might have a 2D input to a neural network that has 64x64 pixels.  This would result in 4,096 input neurons.  This network is either 2D or 4,096D, depending on which set of dimensions you are talking about!

# ### Classification or Regression
# 
# Like many models, neural networks can function in classification or regression:
# 
# * **Regression** - You expect a number as your neural network's prediction.
# * **Classification** - You expect a class/category as your neural network's prediction.
# 
# The following shows a classification and regression neural network:
# 
# ![Neural Network Classification and Regression](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_ann_class_reg.png "Neural Network Classification and Regression")
# 
# Notice that the output of the regression neural network is numeric and the output of the classification is a class.  Regression, or two-class classification, networks always have a single output.  Classification neural networks have an output neuron for each class. 
# 
# The following diagram shows a typical neural network:
# 
# ![Feedforward Neural Networks](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_ann.png "Feedforward Neural Networks")
# 
# There are usually four types of neurons in a neural network:
# 
# * **Input Neurons** - Each input neuron is mapped to one element in the feature vector.
# * **Hidden Neurons** - Hidden neurons allow the neural network to abstract and process the input into the output.
# * **Output Neurons** - Each output neuron calculates one part of the output.
# * **Context Neurons** - Holds state between calls to the neural network to predict.
# * **Bias Neurons** - Work similar to the y-intercept of a linear equation.  
# 
# These neurons are grouped into layers:
# 
# * **Input Layer** - The input layer accepts feature vectors from the dataset.  Input layers usually have a bias neuron.
# * **Output Layer** - The output from the neural network.  The output layer does not have a bias neuron.
# * **Hidden Layers** - Layers that occur between the input and output layers.  Each hidden layer will usually have a bias neuron.
# 
# 

# ### Neuron Calculation
# 
# The output from a single neuron is calculated according to the following formula:
# 
# $ f(x,\theta) = \phi(\sum_i(\theta_i \cdot x_i)) $
# 
# The input vector ($x$) represents the feature vector and the vector $\theta$ (theta) represents the weights. To account for the bias neuron, a value of 1 is always appended to the end of the input feature vector.  This causes the last weight to be interpreted as a bias value that is simply added to the summation. The $\phi$ (phi) is the transfer/activation function. 
# 
# Consider using the above equation to calculate the output from the following neuron:
# 
# ![Single Neuron](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_abstract_nn.png "Single Neuron")
# 
# The above neuron has 2 inputs plus the bias as a third.  This neuron might accept the following input feature vector:
# 
# ```
# [1,2]
# ```
# 
# To account for the bias neuron, a 1 is appended, as follows:
# 
# ```
# [1,2,1]
# ```
# 
# The weights for a 3-input layer (2 real inputs + bias) will always have an additional weight, for the bias.  A weight vector might be:
# 
# ```
# [ 0.1, 0.2, 0.3]
# ```
# 
# To calculate the summation, perform the following:
# 
# ```
# 0.1*1 + 0.2*2 + 0.3*1 = 0.8
# ```
# 
# The value of 0.8 is passed to the $\phi$ (phi) function, which represents the activation function.
# 
# 

# ### Activation Functions
# 
# Activation functions, also known as transfer functions, are used to calculate the output of each layer of a neural network.  Historically neural networks have used a hyperbolic tangent, sigmoid/logistic, or linear activation function.  However, modern deep neural networks primarily make use of the following activation functions:
# 
# * **Rectified Linear Unit (ReLU)** - Used for the output of hidden layers.
# * **Softmax** - Used for the output of classification neural networks. [Softmax Example](http://www.heatonresearch.com/aifh/vol3/softmax.html)
# * **Linear** - Used for the output of regression neural networks (or 2-class classification).
# 
# The ReLU function is calculated as follows:
# 
# $ \phi(x) = \max(0, x) $
# 
# The Softmax is calculated as follows:
# 
# $ \phi_i(z) = \frac{e^{z_i}}{\sum\limits_{j \in group}e^{z_j}} $
# 
# The Softmax activation function is only useful with more than one output neuron.  It ensures that all of the output neurons sum to 1.0.  This makes it very useful for classification where it shows the probability of each of the classes as being the correct choice.
# 
# To experiment with the Softmax, click [here](http://www.heatonresearch.com/aifh/vol3/softmax.html).
# 
# The linear activation function is essentially no activation function:
# 
# $ \phi(x) = x $
# 
# For regression problems, this is the activation function of choice.  
# 
# 

# ### Why ReLU?
# 
# Why is the ReLU activation function so popular?  It was one of the key improvements to neural networks that makes deep learning work. Prior to deep learning, the sigmoid activation function was very common:
# 
# $ \phi(x) = \frac{1}{1 + e^{-x}} $
# 
# The graph of the sigmoid function is shown here:
# 
# ![Sigmoid Activation Function](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_sigmoid.png "Sigmoid Activation Function")
# 
# Neural networks are often trained using gradient descent.  To make use of gradient descent, it is necessary to take the derivative of the activation function.  This allows the partial derivatives of each of the weights to be calculated with respect to the error function.  A derivative is the instantaneous rate of change:
# 
# ![Derivative](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_deriv.png "Derivative")
# 
# The derivative of the sigmoid function is given here:
# 
# $ \phi'(x)=\phi(x)(1-\phi(x)) $
# 
# This derivative is often given in other forms.  The above form is used for computational efficiency. To see how this derivative was taken, see [this](http://www.heatonresearch.com/aifh/vol3/deriv_sigmoid.html).
# 
# The graph of the sigmoid derivative is given here:
# 
# ![Sigmoid Derivative](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_deriv_sigmoid.png "Sigmoid Derivative")
# 
# The derivative quickly saturates to zero as *x* moves from zero.  This is not a problem for the derivative of the ReLU, which is given here:
# 
# $ \phi'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases} $

# ### Why are Bias Neurons Needed?
# 
# The activation functions seen in the previous section specifies the output of a single neuron.  Together, the weight and bias of a neuron shape the output of the activation to produce the desired output.  To see how this process occurs, consider the following equation. It represents a single-input sigmoid activation neural network.
# 
# $ f(x,w,b) = \frac{1}{1 + e^{-(wx+b)}} $ 
# 
# The *x* variable represents the single input to the neural network.  The *w* and *b* variables specify the weight and bias of the neural network.  The above equation is a combination of the weighted sum of the inputs and the sigmoid activation function.  For this section, we will consider the sigmoid function because it clearly demonstrates the effect that a bias neuron has.
# 
# The weights of the neuron allow you to adjust the slope or shape of the activation function.  The following figure shows the effect on the output of the sigmoid activation function if the weight is varied:
# 
# ![Adjusting Weight](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_bias_weight.png "Bias 1")
# 
# The above diagram shows several sigmoid curves using the following parameters:
# 
# ```
# f(x,0.5,0.0)
# f(x,1.0,0.0)
# f(x,1.5,0.0)
# f(x,2.0,0.0)
# ```
# 
# To produce the curves, we did not use bias, which is evident in the third parameter of 0 in each case. Using four weight values yields four different sigmoid curves in the above figure. No matter the weight, we always get the same value of 0.5 when x is 0 because all of the curves hit the same point when x is 0.  We might need the neural network to produce other values when the input is near 0.5.  
# 
# Bias does shift the sigmoid curve, which allows values other than 0.5 when x is near 0. The following figure shows the effect of using a weight of 1.0 with several different biases:
# 
# 
# ![Adjusting Bias](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/class_2_bias_value.png "Bias 1")
# 
# The above diagram shows several sigmoid curves with the following parameters:
# 
# ```
# f(x,1.0,1.0)
# f(x,1.0,0.5)
# f(x,1.0,1.5)
# f(x,1.0,2.0)
# ```
# 
# We used a weight of 1.0 for these curves in all cases.  When we utilized several different biases, sigmoid curves shifted to the left or right.  Because all the curves merge together at the top right or bottom left, it is not a complete shift.
# 
# When we put bias and weights together, they produced a curve that created the necessary output from a neuron.  The above curves are the output from only one neuron.  In a complete network, the output from many different neurons will combine to produce complex output patterns.

# # Module 3 Assignment
# 
# You can find the first assignment here: [assignment 3](https://github.com/jeffheaton/t81_558_deep_learning/blob/master/assignments/assignment_yourname_class3.ipynb)

# In[ ]:




