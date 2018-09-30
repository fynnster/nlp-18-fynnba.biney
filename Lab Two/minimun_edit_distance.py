#!/usr/bin/env python
# coding: utf-8

# # Minimal Edit Distance 

# In[1]:


from numpy import *


# In[76]:


def min_edit_distance(source_word, target_word):  
    len_source_word = len(source_word)+ 1
    len_target_word = len(target_word)+ 1
    the_matrix = zeros ((len_source_word, len_target_word))
    
    for x in range(len_source_word):
        the_matrix [x, 0] = x
        
    for y in range(len_target_word):
        the_matrix [0, y] = y
        
#This is 
    for x in range(1, len_source_word):
        for y in range(1, len_target_word):
            #print(source_word[x-1], target_word[y-1])
            if source_word[x-1] == target_word[y-1]:
                the_matrix [x,y] = min(
                    the_matrix[x-1, y] + 1,
                    the_matrix[x, y-1] + 1,
                    the_matrix[x-1, y-1]+ 0)
            else:
               # print(source_word[x-1], target_word[y-1])
                the_matrix [x,y] = min(
                    the_matrix[x-1,y] + 1,
                    the_matrix[x,y-1] + 1,
                    the_matrix[x-1,y-1] + 2)
            
            
                
    print("The Matrix Results")            
    print (the_matrix)
    print ("The minimal edit distance between "+source_word+" and the "+target_word+" is " + str(int(the_matrix[len_source_word - 1, len_target_word - 1]))+".")


# In[77]:


min_edit_distance("execution", "intention")

min_edit_distance("tell", "feel")


# In[ ]:





# In[ ]:





# In[ ]:




