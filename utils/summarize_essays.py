import re
import os
import pickle
import string
import warnings
from string import punctuation
from pprint import pprint
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from utils.calculate_pmi_features import get_data, tokenize_words

warnings.filterwarnings('ignore')


stop_dict = {}
for w in stopwords.words('english'):
    stop_dict[w] = 1

def generate_freqdists(df, stop_dict):
    words = df['TotalEssayTokens'].tolist()
    words = [item for sublist in words for item in sublist]
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w.lower())
             for w in words
             if w not in string.punctuation]

    #ngrams keep stopwords, unigrams remove them
    ngrams = nltk.ngrams(words, 4)
    ngram_freq = nltk.FreqDist(ngrams)
    
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w.lower())
             for w in words
             if w not in string.punctuation and w.lower() not in stop_dict]
    unigram_freq = nltk.FreqDist(words)
    
    
    return unigram_freq, ngram_freq

def categories_from_hypernyms(termlist, num_cats=20, num_terms=50):

    hypterms = []
    hypterms_dict = defaultdict(list)
    for term in termlist:                  # for each term
        s = wn.synsets(term.lower())       # get its nominal synsets
        for syn in s:                      # for each lemma synset
            for hyp in syn.hypernyms():    # It has a list of hypernyms
                hypterms = hypterms + [hyp.name()]      # Extract the hypernym name and add to list
                hypterms_dict[hyp.name()].append(term)  # Extract examples and add them to dict

    hypfd = nltk.FreqDist(hypterms)

    return hypfd, hypterms_dict