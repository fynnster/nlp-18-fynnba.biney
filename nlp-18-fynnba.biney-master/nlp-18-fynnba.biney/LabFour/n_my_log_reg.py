#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  
import random
import re  
from sklearn.ensemble import RandomForestClassifier  
import nltk 
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords  


# In[2]:


def readTXTAndClean():
    file = open("newfile.txt", "r") 
    mass_list=[]

    a_line = file.readlines()
    
    
    for line in a_line:
        #removing all speciel characters, multiple spaces and single characters
        c_line = re.sub("[^a-zA-Z10]", " ", line.lower())
        c_line2 = re.sub(r'\s+', ' ', c_line, flags=re.I)
        c_line2 = re.sub(r'\s+[a-zA-Z]\s+', ' ', c_line2)
        c_line2 = re.sub(r"[\n,.-/]","", c_line2)
        
        #create final mass list of strings(features)
        mass_list.append(c_line2)

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
        
    return features, classes


# In[24]:


def log_reg(features, classes): 
    
    #this selects 80% of the traning data, uses words that appear atleast 3 times
    #and then converts the features into their number equivalent
    vectorizer = CountVectorizer(max_features=3000, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))  
    X = vectorizer.fit_transform(features).toarray() 
      
    tfidfconverter = TfidfVectorizer(max_features=3000, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))  
    X = tfidfconverter.fit_transform(features).toarray()
          
    #this plits the data into training and testing data    
    #X_train, X_test, y_train, y_test = train_test_split(X, classes, test_size=0.2, random_state=0)  
    
    #this uses the RandomForestClassifier to create a data model(a logistic regression)
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
    classifier.fit(X, classes) 
    
    #this tests the model
    docs_new = []
    file = open("test_sentences.txt", "r")
    a_line = file.readlines()

    for line in a_line:
        docs_new.append(line)
    
    new_X = vectorizer.fit_transform(docs_new).toarray()  
    new_X = tfidfconverter.transform(docs_new).toarray()  
    
    
    ######
  
    
    #this then runs the test data
    y_pred = classifier.predict(new_X) 
    #print(y_pred)
    

    #print(confusion_matrix(classes,y_pred))  
    #print(classification_report(classes,y_pred))  
    #print("The accuracy score is ",accuracy_score(classes, y_pred))
    
    return y_pred    


# In[23]:


def n_myLogReg():
    massList = readTXTAndClean()
    features, classes = masslist_split(massList)
    y_pred = log_reg(features, classes)
    
    
    with open("results_n_myLogReg.txt", "w+") as file:
        file.write(str(y_pred))

n_myLogReg()    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




