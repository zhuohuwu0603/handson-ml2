#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 14: Other Neural Network Techniques**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 14 Video Material
# 
# * Part 14.1: What is AutoML [[Video]](https://www.youtube.com/watch?v=TFUysIR5AB0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_01_automl.ipynb)
# * Part 14.2: Using Denoising AutoEncoders in Keras [[Video]](https://www.youtube.com/watch?v=4bTSu6_fucc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_02_auto_encode.ipynb)
# * Part 14.3: Training an Intrusion Detection System with KDD99 [[Video]](https://www.youtube.com/watch?v=1ySn6h2A68I&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_03_anomaly.ipynb)
# * Part 14.4: Anomaly Detection in Keras [[Video]](https://www.youtube.com/watch?v=VgyKQ5MTDFc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_14_04_ids_kdd99.ipynb)
# * **Part 14.5: The Deep Learning Technologies I am Excited About** [[Video]]() [[Notebook]](t81_558_class_14_05_new_tech.ipynb)
# 
# 

# # Part 14.5: New Technologies
# 
# This course changes often to keep up with the rapidly evolving landscape that is deep learning.  If you would like to continue to monitor this class, I suggest following me on the following:
# 
# * [GitHub](https://github.com/jeffheaton) - I post all changes to GitHub.
# * [Jeff Heaton's YouTube Channel](https://www.youtube.com/user/HeatonResearch) - I add new videos for this class at my channel.
# 
# Currently, there are four technologies that are particularly on my radar for possible future inclusion in this course:
# 
# * Neural Structured Learning (NSL)
# * Bert, AlBert, and Other NLP Technologies
# * Explainability Frameworks
# 
# These section seeks only to provide a high-level overview of these emerging technologies. Complete source code examples are not provided.  Links to supplemental material and code are provided in each subsection. These technologies are described in the following sections.
# 
# # Neural Structured Learning (NSL)
# 
# [Neural Structured Learning (NSL)](https://www.tensorflow.org/neural_structured_learning) provides additional training information to the neural network. [[Cite:bui2018neural]](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46568.pdf) This training information is in the form of a graph that relates individual training cases (rows in your training set) among each other. This allows the neural network to be trained to greater accuracy with a smaller number of labeled data.  When the neural network is ultimately used for scoring and prediction, once training completes, the graph data is no longer provided.
# 
# There are two primary sources that this graph data comes from:
# 
# * Existing Graph Relationships in Data
# * Automatic Adversarial Modification of Images
# 
# Often existing graph relationships may exist in data beyond just the labels that describe what individual data items are.  Consider many photo collections.  There may be collections of images placed into specific albums.  This album placement can form additional training information beyond the actual image data and labels.
# 
# Sometimes graph data cannot be directly obtained for a data set.  This does not necessarily mean that NSL cannot be used.  In such cases, adversarial-like modifications can be made to the data.  Additional examples can be introduced and linked to the original images in the training set.  This might make the final trained neural network more resilient to adversarial example attacks.
# 
# Built into TF 2.0, supports any type of ANN.
# 
# ```
# pip install neural_structured_learning
# ```
# 
# The following figure is from the origional NSL paper. [[Cite:bui2018neural]](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46568.pdf) 
# 
# ![Neural Structured Learning (NSL)](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/nn_graph_input.png)
# 
# A: An example of a graph and feature inputs. In this case, there are two labeled nodes ($x_i$, $x_j$) and one unlabeled
# node ($x_k$), and two edges. The feature vectors, one for each node, are used as neural network inputs. 
# 
# B, C and D: Illustration of Neural Graph Machine for feed-forward, convolution and recurrent networks respectively: the training flow ensures the neural net to make accurate node-level predictions and biases the hidden representations/embeddings of neighboring nodes to be similar. In this example, we force $h_i$ and $h_j$ to be similar as there is an edge connecting $x_i$ and $x_j$ nodes. 
# 
# E: Illustration of how we can construct inputs to the neural network using the adjacency matrix. In this example, we have three nodes and two edges. The feature vector created for each node (shown on the right) has 1’s at its index and indices of nodes that it’s adjacent to.
# 
# The following figure shows that NSL can help when there are fewer training elements, it is from a TensorFlow [presentation by Google](https://www.youtube.com/watch?v=2Ucq7a8CY94). This video is a great starting point if you wish to do more with NSL.
# 
# ![NSL Results](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/neural-graph-results.png)
# 
# The x-axis shows the amount of training data available and the y-axis shows the accuracy attained by the model.  The two graphs show NSL being applied to to different neural network architectures.  NSL can be used with most supervised neural network architectures.  As the number of training elements decreases, there is a period where NSL helps keep the accuracy higher.
# 
# 
# # Bert, AlBert, and Other NLP Technologies
# 
# Natural Language Processing (NLP) has seen a tremendous number of advances in the past few years.  One recent technology is Bidirectional Encoder Representations from
# Transformers (BERT). [[Cite:devlin2018bert]](https://arxiv.org/pdf/1810.04805.pdf) BERT achieved "state of the art" results in the following key NLP benchmarks:
# 
# * [GLUE](https://gluebenchmark.com/) [[Cite:wang2018glue]](https://arxiv.org/abs/1804.07461)
# * [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/) [[Cite:williams2017broad]](https://www.nyu.edu/projects/bowman/multinli/paper.pdf)
# * [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)  [[Cite:rajpurkar2016squad]](https://nlp.stanford.edu/pubs/rajpurkar2018squad.pdf)
# 
# When a framework, such as BERT claims "state of the art" results, it is important to understand what is meant by that.  Consider the GLUE benchmark, it is made up of the following parts:
# 
# * Corpus of Linguistic Acceptability (CoLA)
# * Stanford Sentiment Treebank (SST-2)
# * Microsoft Research Paraphrase Corpus (MRPC)	
# * Semantic Textual Similarity Benchmark (STS-B)
# * Quora Question Pairs (QQP)	
# * Multi-Genre Natural Language Inference Corpus matched/mismatched (MNLI-m/MNLI-mm)
# * Stanford Question Answering Dataset (QNLI)
# * Recognizing Textual Entailment (RTE)
# * Winograd Schema Challenge (WNLI)
# 
# **Single Sentence Tasks**
# 
# CoLA and SST-2 are both single sentence tasks.  CoLA is made up of sample sentences from English grammar textbooks where the authors demonstrated acceptable and unacceptable usage of English grammar.  The task in CoLA is to classify a sentence as acceptable or unacceptable.  Examples include:
# 
# * Acceptable: The angrier Sue gets, the more Fred admires her.
# * Unacceptable: The most you want, the least you eat.
# 
# The task SST-2 is used to analyze sentiment.  Sentences are classified by their degree of positivity vs negativity.  For example:
# 
# * Positive: the greatest musicians
# * Negative: lend some dignity to a dumb story
# 
# **Multi-Sentence Similarity Tasks**
# 
# The MRPC, QQP, and STS-B tasks look at similarity and paraphrase tasks.  The MRPC tests the AIs ability to paraphrase.  Each row contains two sentences and a target that indicates whether each pair captures a paraphrase/semantic equivalence relationship. For example, the following two sentences are considered to be equivalent:
# 
# * He told The Sun newspaper that Mr. Hussein's daughters had British schools and hospitals in mind when they decided to ask for asylum .	
# * "Saddam's daughters had British schools and hospitals in mind when they decided to ask for asylum -- especially the schools," he told The Sun.
# 
# Conversely, though the following two sentences look similar, they are not considered equivalent:
# 
# * Gyorgy Heizler, head of the local disaster unit, said the coach was carrying 38 passengers.	
# * The head of the local disaster unit, Gyorgy Heizler, said the coach driver had failed to heed red stop lights.
# 
# The QQP tasks look at if two questions are asking the same thing.  This data was provided by the Quora website.  Examples of sentences that are considered to ask the same question:
# 
# * What are the coolest Android hacks and tricks you know?	
# * What are some cool hacks for Android phones?
# 
# Similarly, the following two questions are considered to be different in the QQP dataset.
# 
# * If you received a check from Donald Knuth, what did you do and why did you get it?
# * How can I contact Donald Knuth?
# 
# The STS-B dataset evaluates how similar two sentences are. If the target/label it 0, then the two sentences are completely dissimilar.  A target value of 5 indicates that the two sentences are completely equivalent, as they
# mean the same thing.  For example:
# 
# Two sentences with a label of 0:
# 
# * A woman is dancing.	
# * A man is talking.
# 
# Two sentences with a label of 5:
# 
# * A plane is taking off.	
# * An air plane is taking off.
# 
# **Inference Tasks**
# 
# The tasks MNLI, QNLI, RTE, and WNLI are all inference tasks.  The MNLI task provides two sentences that must be labeled neutral, contradiction, or entailment.  For example, the following two sentences are a contradiction:
# 
# * At the end of Rue des Francs-Bourgeois is what many consider to be the city's most handsome residential square, the Place des Vosges, with its stone and red brick facades.
# * Place des Vosges is constructed entirely of gray marble.
# 
# These two sentences are an entailment:
# 
# * I burst through a set of cabin doors, and fell to the ground-	
# * I burst through the doors and fell down.	
# 
# These two sentences are neutral:
# 
# * It's not that the questions they asked weren't interesting or legitimate (though most did fall under the category of already asked and answered).	
# * All of the questions were interesting according to a focus group consulted on the subject.	
# 
# The QNLI task poses a question and supporting sentence.  The label states if the question can be answered by the supporting information.  For example, the following two are labeled as "not_entailment":
# 
# * Question: Which missile batteries often have individual launchers several kilometres from one another?	
# * Answer: When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	
# 
# Similarly, the following two sentences are labeled as "entailment":
# 
# * Question: What two things does Popper argue Tarski's theory involves in an evaluation of truth?	
# * Answer: He bases this interpretation on the fact that examples such as the one described above refer to two things: assertions and the facts to which they refer.	
# 
# The RTE task is similar and looks at whether one sentence entails another.  For example, the following two sentences are labeled as "entailment":
# 
# * Lin Piao, after all, was the creator of Mao's "Little Red Book" of quotations.	
# * Lin Piao wrote the "Little Red Book".
# 
# Similarly the following two sentences are labeled as "not_entailment".
# 
# * Oil prices fall back as Yukos oil threat lifted	
# * Oil prices rise.
# 
# The WNLI task checks to see if two sentences agree to a third.  For example, the following two agree:
# 
# * The foxes are getting in at night and attacking the chickens. They have gotten very bold.	
# * The foxes have gotten very bold.
# 
# Similarly, the following two do not agree:
# 
# * Sam pulled up a chair to the piano, but it was broken, so he had to stand instead.	
# * The piano was broken, so he had to stand instead.
# 
# 
# ** BERT High Level Overview **
# 
# BERT makes use of both pretraining and fine tuning before it is ready to be used to evaluate data.  It is important to understand the different roles of these two functions.
# 
# * **Pretraining** - Ideally this is done once per language.  This is the portion of BERT that most will simply obtain from the original BERT model.  These can be [downloaded here](https://github.com/google-research/bert).
# * **Fine Tuning** - This is where additional layers are added to the base BERT models to adapt it to the intended task.
# 
# The pretraining and fine tuning phases of BERT usage are summarized in the following Figure.
# 
# ![NSL Results](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/bert-1.png)
# 
# Sentences are presented to BERT in the method demonstrated in the following figure.
# 
# ![NSL Results](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/bert-2.png)
# 
# 
# 
# # Explainability Frameworks
# 
# Neural networks are notorious as black box models.  Such model may make accurate predictions; however, explanations of why the black box model chose what it did can be elusive. There are two explainability libraries that I occasionally make use of.
# 
# * [Lime](https://github.com/marcotcr/lime)
# * [Explain it to Me Like I'm 5 (ELI5)](https://eli5.readthedocs.io/en/latest/)
# 

# In[ ]:




