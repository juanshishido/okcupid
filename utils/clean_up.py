import re
import string

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from utils import happyfuntokenizing


def clean_up(input_df, col_names, min_words=5):
    '''
    Input : data frame and list of columns to clean up
    Returns: cleaned data frame (overwrites those columns)
    Drops user if any essay has < min_words number of words (default = 5)
    '''
    assert isinstance(col_names, list), 'Must be type list'
    assert isinstance(input_df, pd.DataFrame), 'Must be pd.DataFrame'
    dfs = []
    for c in col_names:
        df = input_df.copy()
        df[c] = df[c].replace(np.nan, '' , regex=True) \
                     .apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))\
                     .apply(lambda x: re.sub(r"(?:\@|https?\://)\S+", "", x))\
                     .apply(lambda x: re.sub('[.]{2,}', '. ', x))\
                     .apply(lambda x: re.sub('[-]{2,}', ' - ', x))\
                     .apply(lambda x: re.sub('\s+', ' ', x).strip())
        token_count = df[c].str.split().str.len() 
        df = df[token_count > min_words] #drop rows where current essay has < min_words
        df.fillna('', inplace=True)
        dfs.append(df)
    if len(col_names) == 1:
        return dfs[0]
    else:
        return tuple(dfs)

def col_to_data_matrix(df, col_name, remove_stopwords=False,
                       tokenizer=happyfuntokenizing.Tokenizer().tokenize,
                       ngram_range=(1, 3), min_df=0.01):
    '''
    Tokenize and vectorize for a single essay column (given by col_name)
    Returnss two matrices (countvect and tfidf) from that column
    Creates and uses a vocabulary that removes redundancies (ie if a unigram is included in a trigram it is excluded)
    '''
    if remove_stopwords:
        stop_punct = list(string.punctuation) + list(ENGLISH_STOP_WORDS) 
    else:
        stop_punct = list(string.punctuation)

    count_vect = CountVectorizer(stop_words=stop_punct, tokenizer=tokenizer,
                                 ngram_range=ngram_range, min_df=min_df)

    count_vect.fit(df[col_name])
    vocab = count_vect.get_feature_names()
    trigrams = []
    bigrams = []
    unigrams = []
    for v in vocab: #sort in descending order (trigram, bigram, unigram)
        v_set = v.split()
        if len(v_set) == 3:
            trigrams.append(v)
        elif len(v_set) == 2:
            bigrams.append(v)
        else:
            unigrams.append(v)
    vocab2 = trigrams + bigrams + unigrams
    new_vocab = []
    print()
    for v in vocab2:
        v_set = set(v.split())
        if len(v_set) == 3: #first, create list of all trigrams
            new_vocab.append(v)
            continue
        add = True
        for w in new_vocab: #for each bigram
            w_set = set(w.split())
            if v_set <= w_set: #check if bigram is included in the trigrams
                add = False #if so, go to next word
                break
        if add:
            new_vocab.append(v) #if bigram is not in trigram, add to list
    
    count_vect = CountVectorizer(stop_words=stop_punct, tokenizer=tokenizer,
                                 ngram_range=ngram_range, vocabulary=new_vocab) #count vector with new, cleaned vocab
    
    count_matrix = count_vect.fit_transform(df[col_name])
    
    tfidf = TfidfTransformer()
    tfidf_matrix = tfidf.fit_transform(count_matrix)

    return count_matrix, tfidf_matrix, new_vocab
