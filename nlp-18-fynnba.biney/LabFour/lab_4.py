#!/usr/bin/env python
# coding: utf-8

# In[17]:


from n_my_naive_bayes import *
from u_my_naive_bayes import *
from n_my_log_reg import *
from u_my_log_reg import *

import sys
print (sys.argv)

def merge_files():
    filenames = ["imdb_labelled.txt", "amazon_cells_labelled.txt", "yelp_labelled.txt"]
    with open('newfile.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

def main():
    control ="yes" 
    while control =="yes":
        classifier = input("Which classifier do you want to use? nb or lr")  # Python 3
        version = input("Which Version? n or u") 

        if classifier=="nb" and version=="n":
            n_myNaiveBayes()   
        elif classifier=="nb" and version=="u":
            u_myNaiveBayes()
        elif classifier=="lr" and version=="n":
            u_myLogReg()
        elif classifier=="lr" and version=="u":
            u_myLogReg()
        control = input("To continue type yes, else type no")
    
main()
    


# In[ ]:





# In[ ]:





# In[ ]:




