#!/usr/bin/env python
# coding: utf-8

# # NAIVE BAYES CLASSIFIER

# These are the libraries used in the classifier.

# In[21]:


import random
import re
from math import *


# This block of code reads data from a text file, clean the data from the file and shuffle the sentences in the file. 

# In[22]:


def readTXTAndClean():
    file = open("imdb_labelled.txt", "r") 
    mass_list=[]

    a_line = file.readlines()
    
    for line in a_line:
        c_line = re.sub(r"[\n\t!';:£*():?%$#+]", "", line.lower())
        mass_list.append(re.sub(r"[\n,.-/]","", c_line))

    random.shuffle(mass_list)
    file.close()
    return mass_list


# This function splits the file into testdata and training data

# In[23]:


def split_Data(massList):
    split1= round(len(massList)*0.8)
    split2= round(len(massList)*0.2)
    
    trainingData= massList[slice(0,split1)]
    testingData= massList[slice(0,split2)]

    return trainingData, testingData


# The sort_List fucntion sorts thadata into two classes, creates a bag of words, counts the totalnumber of sentences and the number of sentences per class.

# In[15]:


def sort_List(trainingData):
    count= 0
    nOfGSent=0
    nOfBSent=0
    goodClass=[]
    badClass=[]
    
    
    
    for item in trainingData:
        classOfSentence = trainingData[count][-1]
        theWords = trainingData[count][0:-2].split()
        count = count + 1

        if classOfSentence == '1':
            nOfGSent +=1
            goodClass = goodClass + theWords
        else:
            nOfBSent +=1
            badClass = badClass + theWords
            
    goodClass =sorted(goodClass)
    badClass =sorted(badClass)
    
    goodClassDict = {x:goodClass.count(x) for x in goodClass}
    badClassDict = {x:badClass.count(x) for x in badClass}
    bagOfWords = goodClassDict.copy()
    bagOfWords.update(badClassDict)
    
 
    return goodClassDict, badClassDict, bagOfWords, nOfBSent, nOfBSent, count


# The logprior function calculates the logprior per class using;
# 

# $$\log \frac{N_c}{N_{doc}}$$

# In[16]:


def logPior(noofgoodSent, noofbadSent, noofSent):    
    logpriorGood= noofgoodSent/noofSent
    logpriorBad= noofbadSent/noofSent

    
    return logpriorGood, logpriorBad


# In[17]:


def baseloglikelyhood(the_Class,bagOfWords):
    loglikelyhood = 0
    countVocab= len(bagOfWords)
    total_words_per_class= sum(the_Class.values())
    
    denominator= countVocab+ total_words_per_class
    
    
    for key, value in bagOfWords.items():
        if key in the_Class.keys():
            count= the_Class[key]+1
            loglikelyhood = log(count/denominator)
        else:
            count= 1
            loglikelyhood = log(count/denominator)
        
            
    return float(loglikelyhood)
    


# The loglikelyhood function uses the baseloglikelyhood funtion to calculate the log likelyhood per word.
# It returns two dictionaries, likelyhoodgood, likelyhoodbad using;

# $$\log \frac{count(w,c)+1}{\sum_{w\in v}(count(w′,c)+1)}$$

# In[18]:


def loglikelyhood(goodDict, badDict, wordBag):
    likelihood_dict={}
    likelyhoodgood=[]
    likelyhoodbad=[]
    words=[]
    for key, value in wordBag.items():
        words.append(key)
        likelyhoodgood.append(baseloglikelyhood(goodDict,wordBag))
        likelyhoodbad.append(baseloglikelyhood(badDict,wordBag))
        
    likelyhoodgood = dict(zip(words, likelyhoodgood))
    likelyhoodbad = dict(zip(words, likelyhoodbad))
    
    return likelyhoodgood, likelyhoodbad
    


# In[19]:


#def test_naive_Bayes():


# In[20]:


def main():
    
    #this trains the Naive Bayes Classifier
    massList = readTXTAndClean()
    trainingData, testingData = split_Data(massList)
    goodDict, badDict, wordBag, noofgoodSent, noofbadSent, noofSent  = sort_List(trainingData)
    logprobofGoodwords, logprobofBadwords = logPior(noofgoodSent, noofbadSent, noofSent)
    logProbs = loglikelyhood(goodDict, badDict, wordBag)
    
    #this tests the 
main()


# In[ ]:





# In[ ]:





# In[ ]:




