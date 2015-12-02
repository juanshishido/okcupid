import re
import pickle
import string

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from utils.hash import *
from utils import happyfuntokenizing


#Gets done in a reasonable time, joins all the essays together into one long text
#Also gets rid of HTML tags using the BeautifulSoup Library
def get_data():
    df = pd.read_csv('data/profiles.20120630.csv')

    def remove_nan(s):
        if type(s) == float:
            return ''
        return s
    
    #dealing with the nan values of essays
    essays = df.columns.values[7:17]
    for text in essays:
        df[text] = df[text].apply(remove_nan)
    df['TotalEssays'] = df[essays].apply(lambda x: ' '.join(x), axis=1)
    df['TotalEssays'] = df['TotalEssays'].apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))\
                                         .apply(lambda x: re.sub('\s+', ' ', x).strip())
    return df[df.TotalEssays.str.len() > 0]

#Tokenizes the concatenated essay using the hft
def tokenize_words(df):
    tokenizer = happyfuntokenizing.Tokenizer()
    df['TotalEssayTokens'] = df['TotalEssays'].apply(lambda x: tokenizer.tokenize(x))
    return df

#Generates freqdists for unigrams, bigrams, and trigrams
def generate_freqdists(df, stop_dict):
    words = df['TotalEssayTokens'].tolist()
    words = [item for sublist in words for item in sublist]
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w.lower())
             for w in words
             if w not in string.punctuation and w.lower() not in stop_dict]
    unigram_freq = nltk.FreqDist(words)
    
    bigrams = nltk.ngrams(words, 2)
    bigram_freq = nltk.FreqDist(bigrams)
    
    trigrams = nltk.ngrams(words, 3)
    trigram_freq = nltk.FreqDist(trigrams)
    
    return unigram_freq, bigram_freq, trigram_freq

#Generates a vocabularly for the count vectorizer by computing the pmi for bigrams and trigrams like the paper
def generate_vocab(freq_dists):
    vocab = []
    unigram_freq = freq_dists[0]
    for k in unigram_freq.keys():
        if unigram_freq[k] >= 3:
            vocab.append(k)
    
    vocab2 = []
    bigram_freq = freq_dists[1]
    unicount = sum(unigram_freq.values())
    bicount = sum(bigram_freq.values())
    for k in bigram_freq.keys():
        num = bigram_freq[k] / bicount
        denom = (unigram_freq[k[0]] / unicount) * (unigram_freq[k[1]] / unicount)
        if num / denom > 4:
            vocab2.append(k)
    
    vocab3 = []
    trigram_freq = freq_dists[2]
    tricount = sum(trigram_freq.values())
    for k in trigram_freq.keys():
        num = trigram_freq[k] / tricount
        denom = (unigram_freq[k[0]] / unicount) * (unigram_freq[k[1]] / unicount) * (unigram_freq[k[2]] / unicount)
        if num / denom > 6:
            vocab3.append(k)

    return set(vocab + vocab2 + vocab3)

#Filters out words from the vocabulary v that are used by less than 1% of the users in the dataframe
def filter_vocab(df, v, stop_dict):
    #Every word gets a dictionary entry
    v_dict = {}
    for w in v:
        v_dict[w] = 0
    wordnet_lemmatizer = WordNetLemmatizer()
    
    #Going through every essay's unigrams, bigrams, trigrams
    #only adding to dictionary once if they exist in that essay
    essays = df['TotalEssayTokens'].tolist()
    for i,e in enumerate(essays):
        words = [wordnet_lemmatizer.lemmatize(w.lower())
                 for w in e
                 if w not in string.punctuation and w.lower() not in stop_dict]
        uni = words
        bi = nltk.ngrams(words, 2)
        tri = nltk.ngrams(words, 3)
        total = set(list(uni) + list(bi) + list(tri)) #only counting once
        for t in total:
            if t in v_dict:
                v_dict[t] += 1
    #Removing words that occur in less than 599 different users           
    final_vocab = []
    for k in v_dict.keys():
        if v_dict[k] >= 599:
            if type(k) == tuple:
                final_vocab.append(' '.join(k)) #joining bigrams and trigrams
            else:
                final_vocab.append(k)
    return final_vocab

#Creates the data matrix, normalizes by user word count, writes the matrix to a pickle file just in case
#returns data matrix if needed.
def create_data_matrix(df, vocab, filename, ppath, force_update=False):
    
    count_vect = CountVectorizer(stop_words='english',
                                 tokenizer=happyfuntokenizing.Tokenizer().tokenize,
                                 ngram_range=(1, 3), analyzer='word', vocabulary=vocab)
    data_matrix = count_vect.fit_transform(df['TotalEssays'])
    
    data_matrix_dense = data_matrix.todense().astype(float)
    
    #Normalizing each row by the number of words that user uses
    essays = df['TotalEssayTokens'].tolist()
    for i,e in enumerate(essays):
        data_matrix_dense[i, :] = data_matrix_dense[i, :] / len(e)
            
    with open(ppath, 'wb') as f:
        pickle.dump(data_matrix_dense, f)

    #add or update hash
    hash_update(p, hash_get(p), force_update=force_update)
        
    return data_matrix

def run_PMI(p, ppath, force_update):
    stop_dict = {}
    for w in stopwords.words('english'):
        stop_dict[w] = 1

    main_df = get_data()
    main_df = tokenize_words(main_df)
    freq_dists = generate_freqdists(main_df, stop_dict)
    vocab = generate_vocab(freq_dists)
    vocab = filter_vocab(main_df, vocab, stop_dict)
    data_matrix = create_data_matrix(main_df, vocab, p, force_update)
    return data_matrix
