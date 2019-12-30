#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 11: Natural Language Processing and Speech Recognition**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 11 Material
# 
# * **Part 11.1: Getting Started with Spacy in Python** [[Video]](https://www.youtube.com/watch?v=A5BtU9vXzu8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_01_spacy.ipynb)
# * Part 11.2: Word2Vec and Text Classification [[Video]](https://www.youtube.com/watch?v=nWxtRlpObIs&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_02_word2vec.ipynb)
# * Part 11.3: What are Embedding Layers in Keras [[Video]](https://www.youtube.com/watch?v=OuNH5kT-aD0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_03_embedding.ipynb)
# * Part 11.4: Natural Language Processing with Spacy and Keras [[Video]](https://www.youtube.com/watch?v=BKgwjhao5DU&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_11_04_text_nlp.ipynb)
# * Part 11.5: Learning English from Scratch with Keras and TensorFlow [[Video]](https://www.youtube.com/watch?v=Y1khuuSjZzc&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN&index=58) [[Notebook]](t81_558_class_11_05_english_scratch.ipynb)

# # Part 11.1: Getting Started with Spacy in Python
# 
# When neural networks are applied to natural language processing you must decide if you want to operate at the word or character level.  Up to this point we've operated primarily at the character level.  This was the case for the Treasure Island text pirate story generator.  We used word-level NLP for the image caption generator.  In this module, the focus will be primarily upon word-level NLP.  Particularly, we will examine some of the NLP tools that can be used to process words before they are sent to the neural network.  There are two very common NLP libraries for Python:
# 
# * [NLTK](https://www.nltk.org/)
# * [Spacy](https://spacy.io/)
# 
# In this course we will focus on Spacy.  I prefer spacy because of the nice object abstraction of sentences that it provides.  However, both are fine libraries.
# 
# ### Installing Spacy
# 
# Spacy can be installed with a simple PIP install. This was included in the list of packages to install for this course.  You will need to ensure that you've installed a language with Spacy.  If you do not, you will get the following error:
# 
# ```
# OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
# ```
# 
# To install English, use the following command:
# 
# ```
# python -m spacy download en
# ```
# 
# 

# ### Tokenization
# 
# Tokenization. Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called tokens , perhaps at the same time throwing away certain characters, such as punctuation.  Consider how the following sentences might be broken into words.
# 
# * This is a test.
# * Ok, but what about this?
# * Is USA the same as U.S.A.?
# * What is the best data-set to use?
# * I think I will do this-no wait, I will do that.

# In[1]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"Apple is looking at buying a U.K. startup for $1 billion")
for token in doc:
    print(token.text)


# In[2]:


for word in doc:  
    print(word.text,  word.pos_)


# In[3]:


for word in doc:
    print(f"{word} is like number? {word.like_num}")


# ### Sentence Diagramming

# In[4]:


import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(u"I want an iPad, Laptop, and a dog.")
displacy.serve(doc, style="dep")


# **Note, you will have to manually stop the above cell**

# In[5]:


print(doc)


# In[6]:


import spacy

# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

sentence = "The striped bats are hanging on their feet for best"

# Parse the sentence using the loaded 'en' model object `nlp`
doc = nlp(sentence)

# Extract the lemma for each token and join
" ".join([token.lemma_ for token in doc])
#> 'the strip bat be hang on -PRON- foot for good'


# ### Stop Words
# 
# 

# In[7]:


from spacy.lang.en.stop_words import STOP_WORDS

print(STOP_WORDS)

