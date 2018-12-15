#!/usr/bin/env python
# coding: utf-8

# # NAIVE BAYES CLASSIFIER

# These are the libraries used in the classifier.

# In[1]:


import random
import re
from math import *


# In[2]:


def merge_files():
    filenames = ["imdb_labelled.txt", "amazon_cells_labelled.txt", "yelp_labelled.txt"]
    with open('newfile.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())


# This block of code reads data from a text file, clean the data from the file and shuffle the sentences in the file. 

# In[3]:


def readTXTAndClean():
    file = open("newfile.txt", "r") 
    mass_list=[]

    a_line = file.readlines()
    
    for line in a_line:
        c_line = re.sub(r"[\n\t!';:£*():?%$#+]", "", line.lower())
        mass_list.append(re.sub(r"[\n,.-/]","", c_line))
    

    random.shuffle(mass_list)
    file.close()
    return mass_list


# This function splits the file into testdata and training data

# In[4]:


def split_Data(massList):
    split1= round(len(massList)*.8)
    split2= round(len(massList)*.2)
    
    trainingData= massList[slice(0,split1)]
    testingData= massList[slice(0,split2)]

    return trainingData, testingData


# The sort_List fucntion sorts thadata into two classes, creates a bag of words, counts the totalnumber of sentences and the number of sentences per class.

# In[5]:


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
    
    #print(bagOfWords)
    

    return goodClassDict, badClassDict, bagOfWords, nOfGSent, nOfBSent, count


# The logprior function calculates the logprior per class using;
# 

# $$\log \frac{N_c}{N_{doc}}$$

# In[6]:


def logPior(noofgoodSent, noofbadSent, noofSent):    
    logpriorGood= log10(noofgoodSent/noofSent)
    logpriorBad= log10(noofbadSent/noofSent)

    print(logpriorGood, logpriorBad)
    return logpriorGood, logpriorBad


# $$\log \frac{count(w,c)+1}{\sum_{w\in v}(count(w′,c)+1)}$$

# In[7]:


def baseloglikelyhood(the_Class,bagOfWords):
    loglikelyhood = 0
    countVocab= len(bagOfWords)
    loglikely= {}
    total_words_per_class= sum(the_Class.values())
#     print(the_Class.values())
    
    denominator = total_words_per_class + countVocab
    
    
    for key, value in bagOfWords.items():
        if key in the_Class.keys():
            count= the_Class[key] + 1
            loglikelyhood = log(count/denominator)
            loglikely[key]= loglikelyhood
            
        else:
            count= 1
            loglikelyhood = log(count/denominator)
            loglikely[key]= loglikelyhood
        
       
    return loglikely
    


# In[8]:


def x(bagOfWords,the_Class, word):
    if word in the_Class.keys():
        numerator = the_Class[word] +1
    else:
        numerator = 1   
    
    
    denominator= len(set(bagOfWords.keys()) - set(the_Class.keys()))+len(the_Class.keys())
    denominator+= sum(the_Class.values())
    
    log_likelihood =  log(numerator/denominator)
    
    
    
    return log_likelihood


# The loglikelyhood function uses the baseloglikelyhood funtion to calculate the log likelyhood per word.
# It returns two dictionaries, likelyhoodgood, likelyhoodbad using;

# In[9]:


def loglikelyhood(goodDict, badDict, wordBag):
    likelyhoodgood={}
    likelyhoodbad={}
    words=[]
    
 
    likelyhoodgood = baseloglikelyhood(goodDict,wordBag)
    likelyhoodbad = baseloglikelyhood(badDict,wordBag)
    
    return likelyhoodgood, likelyhoodbad
    


# In[23]:


def test_naive_Bayes(testClass,logprobofGoodwords, logprobofBadwords, logProbGood,logProbBad ):
    
    outcome=[]
    pos= 0
    neg= 0

    for i in range(len(testClass)):
        a_sentence = testClass[i]
        words = testClass[i][0:-1].split()
        
    
        for j in words: 
            if j in logProbGood.keys():
                pos = logprobofGoodwords + logProbGood[j]
                
        for j in words: 
            if j in logProbBad.keys():
                neg = logprobofBadwords + logProbBad[j]
                
        
        with open("results_n_myLogReg.txt", "w+") as file:
        file.write(outcome)
        
        if (pos > neg):
            outcome.append(1)
            print(1, a_sentence)
        elif (pos < neg):
            outcome.append(0)
            print(0, a_sentence)
        
        with open("results.txt", "w+") as file:
            file.write(str(outcome,a_sentence))

        
    return outcome
    
     


# In[22]:


def runAll():
    merge_files()
   
    #this trains the Naive Bayes Classifier
    massList = readTXTAndClean()
    trainingData, testingData = split_Data(massList)
    goodDict, badDict, wordBag, noofgoodSent, noofbadSent, noofSent  = sort_List(trainingData)
    logprobofGoodwords, logprobofBadwords = logPior(noofgoodSent, noofbadSent, noofSent)
    logProbGood,logProbBad = loglikelyhood(goodDict, badDict, wordBag)
    
    

    test_naive_Bayes(testingData,logprobofGoodwords, logprobofBadwords, logProbGood,logProbBad)
    
runAll()


# In[10]:





# In[ ]:





# In[ ]:





# In[ ]:




