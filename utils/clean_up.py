import re
import string

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from utils import happyfuntokenizing


def clean_up(df, col_names, min_words = 5):
    '''
    Input : data frame and list of columns to clean up
    Returns: cleaned data frame (overwrites those columns)
    Drops user if any essay has < min_words number of words (default = 5)
    '''
    for c in col_names:
        df[c] = df[c].replace(np.nan, '' , regex=True) \
                    .apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))\
                    .replace('\n', ' ')                  \
                    .apply(lambda x: TAG_RE.sub(' ', x)) \
                    .apply(lambda x: re.sub('\s+', ' ', x).strip())
        token_count = df[c].str.split().str.len() 
        df = df[token_count > min_words] #drop rows where current essay has < min_words
    return df

def col_to_data_matrix(df, col_name):
    '''
    Tokenize and vectorize for a single essay column (given by col_name)
    Returnss two matrices (countvect and tfidf) from that column
    '''
    
    count_vect = CountVectorizer(stop_words='english',
                                 tokenizer=happyfuntokenizing.Tokenizer().tokenize,
                                 ngram_range=(1, 3), analyzer='word', min_df = 0.01)

    count_matrix = count_vect.fit_transform(df[col_name])

    vocab = count_vect.get_feature_names()
    
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    return count_matrix, tfidf_matrix, vocab
