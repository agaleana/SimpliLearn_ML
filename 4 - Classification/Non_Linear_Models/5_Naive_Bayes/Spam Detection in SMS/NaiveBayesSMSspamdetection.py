
# coding: utf-8

# # SMS Spam detection using Naive Bayes
# 
# Source : talk on "Data Science with Python" at the [University of Economics](https://www.vse.cz/english/) in Prague, December 2014. [@RadimRehurek](https://twitter.com/radimrehurek).
# 
# The goal of this talk is to demonstrate some high level, introductory concepts behind (text) machine learning. The concepts are demonstrated by concrete code examples in this notebook, which you can run yourself (after installing IPython, see below), on your own computer.
# 
# The talk audience is expected to have some basic programming knowledge (though not necessarily Python) and some basic introductory data mining background. This is *not* an "advanced talk" for machine learning experts.
# 
# The code examples build a working, executable prototype: an app to classify phone SMS messages in English (well, the "SMS kind" of English...) as either "spam" or "ham" (=not spam).

# [![](http://radimrehurek.com/data_science_python/python.png)](http://xkcd.com/353/)

# The language used throughout will be [Python](https://www.python.org/), a general purpose language helpful in all parts of the pipeline: I/O, data wrangling and preprocessing, model training and evaluation. While Python is by no means the only choice, it offers a unique combination of flexibility, ease of development and performance, thanks to its mature scientific computing ecosystem. Its vast, open source ecosystem also avoids the lock-in (and associated bitrot) of any single specific framework or library.
# 
# Python (and of most its libraries) is also platform independent, so you can run this notebook on Windows, Linux or OS X without a change.
# 
# One of the Python tools, the IPython notebook = interactive Python rendered as HTML, you're watching right now. We'll go over other practical tools, widely used in the data science industry, below.

# # End-to-end example: automated spam filtering

# In[20]:


#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
#import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve


# ## Step 1: Load data, look around

# Skipping the *real* first step (fleshing out specs, finding out what is it we want to be doing -- often highly non-trivial in practice!), let's download the dataset we'll be using in this demo. Go to https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection and download the zip file. Unzip it under `data` subdirectory. You should see a file called `SMSSpamCollection`, about 0.5MB in size:

# ```bash
# $ ls -l data
# total 1352
# -rw-r--r--@ 1 kofola  staff  477907 Mar 15  2011 SMSSpamCollection
# -rw-r--r--@ 1 kofola  staff    5868 Apr 18  2011 readme
# -rw-r-----@ 1 kofola  staff  203415 Dec  1 15:30 smsspamcollection.zip
# ```

# This file contains **a collection of more than 5 thousand SMS phone messages** (see the `readme` file for more info):

# In[14]:


messages = [line.rstrip() for line in open('./data/SMSSpamCollection')]
#print (messages.length)


# A collection of texts is also sometimes called "corpus". Let's print the first ten messages in this SMS corpus:

# In[15]:


for message_no, message in enumerate(messages[:10]):
    print (message_no, message)


# We see that this is a [TSV](http://en.wikipedia.org/wiki/Tab-separated_values) ("tab separated values") file, where the first column is a label saying whether the given message is a normal message ("ham") or "spam". The second column is the message itself.
# 
# This corpus will be our labeled training set. Using these ham/spam examples, we'll **train a machine learning model to learn to discriminate between ham/spam automatically**. Then, with a trained model, we'll be able to **classify arbitrary unlabeled messages** as ham or spam.

# [![](http://radimrehurek.com/data_science_python/plot_ML_flow_chart_11.png)](http://www.astroml.org/sklearn_tutorial/general_concepts.html#supervised-learning-model-fit-x-y)

# Instead of parsing TSV (or CSV, or Excel...) files by hand, we can use Python's `pandas` library to do the work for us:

# In[10]:


messages = pandas.read_csv("./data/SMSSpamCollection", sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
print (messages)


# With `pandas`, we can also view aggregate statistics easily:

# In[5]:

print()
print("Aggregate statistics of messages :")
print()
print(messages.groupby('label').describe())
print()


# How long are the messages?

# In[6]:

print("Length of initial few messages :")
print()
messages['length'] = messages['message'].map(lambda text: len(text))
print (messages.head())
print()


# In[7]:


messages.length.plot(bins=20, kind='hist')


# In[8]:

print("Aggregate info about LENGTH of messages :")
print()
print(messages.length.describe())
print()

# What is that super long message?

# In[9]:

print("Print a message longer than 900 characters long :")
print()
print (list(messages.message[messages.length > 900]))
print()


# Is there any difference in message length between spam and ham?

# In[10]:


messages.hist(column='length', by='label', bins=50)


# Good fun, but how do we make computer understand the plain text messages themselves? Or can it under such malformed gibberish at all?

# ## Step 2: Data preprocessing

# In this section we'll massage the raw messages (sequence of characters) into vectors (sequences of numbers).
# 
# The mapping is not 1-to-1; we'll use the [bag-of-words](http://en.wikipedia.org/wiki/Bag-of-words_model) approach, where each unique word in a text will be represented by one number.
# 
# As a first step, let's write a function that will split a message into its individual words:

# In[11]:


def split_into_tokens(message):
#    message = str(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words


# Here are some of the original texts again:
#     

# In[12]:

print("Initial few messages again :")
print()
print(messages.message.head())
print()


# ...and here are the same messages, tokenized:

# In[13]:

print("Tokenized messages :")
print()
print(messages.message.head().apply(split_into_tokens))
print()


# NLP questions:
# 
# 1. Do capital letters carry information?
# 2. Does distinguishing inflected form ("goes" vs. "go") carry information?
# 3. Do interjections, determiners carry information?
# 
# In other words, we want to better "normalize" the text.
# 
# With textblob, we'd detect [part-of-speech (POS)](http://www.ling.upenn.edu/courses/Fall_2007/ling001/penn_treebank_pos.html) tags with:

# In[14]:

print("Word, Part of Speech POS tag for sentence 'Hello world, how is it going?' :")
print()
print(TextBlob("Hello world, how is it going?").tags)  # list of (word, POS) pairs
print()


# and normalize words into their base form ([lemmas](http://en.wikipedia.org/wiki/Lemmatisation)) with:

# In[15]:


def split_into_lemmas(message):
#    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

# CellStrat
# For example, in English, the verb 'to walk' may appear as 'walk', 'walked', 'walks', 'walking'. The base form, 'walk', that one might look up in a dictionary, is called the lemma for the word.
print("After lemmatization, the messages are :")
print()
print(messages.message.head().apply(split_into_lemmas))
print()


# Better. You can probably think of many more ways to improve the preprocessing: decoding HTML entities (those `&amp;` and `&lt;` we saw above); filtering out stop words (pronouns etc); adding more features, such as an word-in-all-caps indicator and so on.

# ## Step 3: Data to vectors

# Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# 
# Doing that requires essentially three steps, in the bag-of-words model:
# 
# 1. counting how many times does a word occur in each message (term frequency)
# 2. weighting the counts, so that frequent tokens get lower weight (inverse document frequency) - in order to eliminate common
# words like "is", "that", "and" etc.
# 3. normalizing the vectors to unit length, to abstract from the original text length (L2 norm)

# Each vector has as many dimensions as there are unique words in the SMS corpus:

# In[16]:

print ("Data to vectors - Convert the messages to a matrix of token counts")
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
print()
#print len(bow_transformer.vocabulary_)


# Here we used `scikit-learn` (`sklearn`), a powerful Python library for teaching machine learning. It contains a multitude of various methods and options.
# 
# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new `bow_transformer`:

# In[17]:

print("take one particular text message :")
print()
message4 = messages['message'][3]
print (message4)
print()


# In[18]:

print("After Bag of Words and conversion to vector, above text message :")
print()
bow4 = bow_transformer.transform([message4])
print (bow4)
print()
print("Shape of Bag of Words :")
print()
print (bow4.shape)
print()


# So, nine unique words in message nr. 4, two of them appear twice, the rest only once. Sanity check: what are these words the appear twice?

# In[19]:

print("Sanity check: what are these words that appear twice :")
print()
#print (bow_transformer.get_feature_names()[6736])
#print (bow_transformer.get_feature_names()[8013])
print (bow_transformer.get_feature_names()[4191])
print (bow_transformer.get_feature_names()[9282])
print()


# The bag-of-words counts for the entire SMS corpus are a large, sparse matrix:

# In[20]:

print ("Using the bag of words transformer, transform all the messages :")
messages_bow = bow_transformer.transform(messages['message'])
print()
print ('sparse matrix shape:', messages_bow.shape)
print ('number of non-zeros:', messages_bow.nnz)
print ('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
print()


# And finally, after the counting, the term weighting and normalization can be done with [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf), using scikit-learn's `TfidfTransformer`:

# In[21]:

print("Now perform TFIDF fit and transform")
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print("After TFIDF fit and transformation, above text message :")
print()
print (tfidf4)
print()


# What is the IDF (inverse document frequency) of the word `"u"`? Of word `"university"`?

# In[22]:

print("What is the IDF (inverse document frequency) of the word 'u'? Of word 'university'?:")
print()
print (tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print (tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
print()


# To transform the entire bag-of-words corpus into TF-IDF corpus at once:

# In[23]:

print("Transform the entire bag-of-words corpus into TF-IDF corpus at once :")
messages_tfidf = tfidf_transformer.transform(messages_bow)
print("After TFIDF transformation, print shape :")
print()
print (messages_tfidf.shape)
print()


# There are a multitude of ways in which data can be proprocessed and vectorized. These two steps, also called "feature engineering", are typically the most time consuming and "unsexy" parts of building a predictive pipeline, but they are very important and require some experience. The trick is to evaluate constantly: analyze model for the errors it makes, improve data cleaning & preprocessing, brainstorm for new features, evaluate...

# ## Step 4: Training a model, detecting spam

# With messages represented as vectors, we can finally train our spam/ham classifier. This part is pretty straightforward, and there are many libraries that realize the training algorithms.

# We'll be using scikit-learn here, choosing the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier to start with:

# In[24]:


#get_ipython().magic("time spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])")
print ("Now perform MultinomialNB fit to get the Spam Detector")
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
print()


# Let's try classifying our single random message:

# In[25]:

print("Let's use this spam detector to try to classify our single random message :")
print()
print ('predicted:', spam_detector.predict(tfidf4)[0])
print ('expected:', messages.label[3])
print()


# Hooray! You can try it with your own texts, too.
# 
# A natural question is to ask, how many messages do we classify correctly overall?

# In[26]:


all_predictions = spam_detector.predict(messages_tfidf)
print("Print all predictions :")
print()
print (all_predictions)
print()


# In[27]:


print ('accuracy of spam predictions', accuracy_score(messages['label'], all_predictions))
print()
print ('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
print ('(row=expected, col=predicted)')
print()


# In[28]:


plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()


# From this confusion matrix, we can compute precision and recall, or their combination (harmonic mean) F1:

# In[29]:

print("From this confusion matrix, we can compute precision and recall, or their combination (harmonic mean) F1")
print()
print (classification_report(messages['label'], all_predictions))
print()