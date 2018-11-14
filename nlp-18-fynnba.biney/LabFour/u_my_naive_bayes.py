#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk  
import numpy as np  
import random
import re  
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[2]:


def readTXTAndClean():
    file = open("newfile.txt", "r") 
    mass_list=[]

    a_line = file.readlines()
    
    
    for line in a_line:
        #removing all speciel characters
        c_line = re.sub("[^a-zA-Z10]", " ", line.lower())
        
        #create final mass list of strings(features)
        mass_list.append(c_line)

    random.shuffle(mass_list)
    file.close()
    return mass_list


# In[3]:


def masslist_split(massList):
    features =[]
    classes =[]

    #to seperate the feature from its actual class
    for i in range(len(massList)):
        feature = massList[i][:-3]
        classe = massList[i][-2:-1]
        
        features.append(feature)
        classes.append(classe)
        
    
    #print(features)    
    return features, classes


# In[4]:


def naive_bayes(features, classes):
    count_vect = CountVectorizer()  
    counts = count_vect.fit_transform(features)  
    
    transformer = TfidfTransformer()
    counts_tf = transformer.fit_transform(counts)   
    
    model = MultinomialNB().fit(counts_tf, classes) 
    
    docs_new = []
    
    file = open("test_sentences.txt", "r")
    a_line = file.readlines()

    for line in a_line:
        docs_new.append(line)
    
    
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = transformer.transform(X_new_counts)

    predicted = model.predict(X_new_tfidf)
    
    return predicted


# In[5]:


def u_myNaiveBayes():
    massList = readTXTAndClean()
    features, classes = masslist_split(massList)
    predicted = naive_bayes(features, classes)
    
    with open("results_u_myNaiveBayes.txt", "w+") as file:
        file.write(str(predicted))


u_myNaiveBayes()    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




