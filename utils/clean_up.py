import re
import pickle
import string

import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer

from utils.hash import *
from utils import happyfuntokenizing


def clean_up(df, col_names, min_words = 5):
    '''
    Input : data frame and list of columns to clean up
    Returns: cleaned data frame (overwrites those columns)
    Drops user if any essay has < min_words number of words (default = 5)
    '''
    for c in col_names:
        df[c] = df[c].apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))\
                .apply(lambda x: re.sub('\s+', ' ', x).strip())
        #drop rows where current essay has < min_words
        token_count = df[c].str.split().str.len()
        df = df[token_count > min_words]

def create_data_matrix(df, col_name, ppath, force_update=False):
    '''
    Tokenize and vectorize for a single essay column (given by col_name)
    Returns a datamatrix from that column
    '''
    
    count_vect = CountVectorizer(stop_words='english',
                                 tokenizer=happyfuntokenizing.Tokenizer().tokenize,
                                 ngram_range=(1, 3), analyzer='word')

    data_matrix = count_vect.fit_transform(df[col_name])
    
    data_matrix_dense = data_matrix.todense().astype(float)
                
    with open(ppath, 'wb') as f:
        pickle.dump(data_matrix_dense, f)

    #add or update hash
    hash_update(p, hash_get(p), force_update=force_update)
        
    return data_matrix

