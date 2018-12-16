#!/usr/bin/env python
# coding: utf-8

# This merges all the files that would be used as imput data for the NAIVE BAYES CLASSIFIER

# In[ ]:


def merge_files():
    filenames = ["imdb_labelled.txt", "amazon_cells_labelled.txt", "yelp_labelled.txt"]
    with open('newfile.txt', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())
                
merge_files()


# In[ ]:




