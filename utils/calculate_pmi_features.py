
# coding: utf-8

# In[1]:

get_ipython().magic(u'reset')


# #Calculating PMI Features
# The basic algorithm is as follows:  
# 1. Read in the data as pandas data frame  
# 2. Combine all the essays together, remove HTML tags, remove newline characters  
# 3. Tokenize all the essays using the happyfuntokenizer  
# 4. Use the tokens to generate unigrams, bigrams, and trigrams. Stopwords are removed for bigrams and trigrams as well as unigrams. I felt that the bigrams and trigrams were more informative after this step. But we can add them back in if you'd like. These are stored in freqdists that come in handy for the PMI calculation.       
# 5. Only keep unigrams that occur more than 3 times, bigrams with PMI > 4, and trigrams with PMI > 6. This starts building the vocabulary that will be used to vectorize the essay data.      
# 6. Filter out unigrams, bigrams, and trigrams that are used by less than 1% of the users.   
# 7. Use the CountVectorizer to vectorize the user data, making sure to remove stopwords, use the hft to tokenize, 
# analyze using 1 to 3-grams, and the new vocabulary. The rows are normalized by the number of words a user uses in all of their essays.  
# 8. The data matrix is written to a pickle file in case you don't want to run this again, but it doesn't take too long. Less than 10 minutes.

# In[1]:

import re
import warnings
import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from utils import *
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup
import string
get_ipython().magic(u'matplotlib inline')
plt.style.use('ggplot')
warnings.filterwarnings('ignore')


# In[2]:

#Checking whether 'w not in stopwords.words('english') takes FOREVER. So I converted it to a dictionary
#because checking whether or not w is in a list of size n takes O(n) time, but looking something up in
#a dictionary takes O(1) time.
stop_dict = {}
for w in stopwords.words('english'):
    stop_dict[w] = 1


# In[3]:

#Gets done in a reasonable time, joins all the essays together into one long text
#Also gets rid of HTML tags using the BeautifulSoup Library
def get_data():
    df = pd.read_csv('data/profiles.20120630.csv')
    #print(df.columns.values )
    def remove_nan(s):
        if type(s) == float:
            return ''
        return s
    
    #dealing with the nan values of essays
    essays = df.columns.values[7:17]
    for text in essays:
        df[text] = df[text].apply(remove_nan)
    df['TotalEssays'] = df[essays].apply(lambda x: ' '.join(x), axis=1)
    df['TotalEssays'] = df['TotalEssays'].apply(lambda x: BeautifulSoup(x).getText().replace('\n', ' '))
                                            .apply(lambda x: re.sub('\s+', ' ', x).strip())
    return df[df.TotalEssays.str.len() > 0]


# In[4]:

#Tokenizes the concatenated essay using the hft
def tokenize_words(df):
    tokenizer = happyfuntokenizing.Tokenizer()
    df['TotalEssayTokens'] = df['TotalEssays'].apply(lambda x: tokenizer.tokenize(x))
    return df


# In[5]:

#Generates freqdists for unigrams, bigrams, and trigrams
def generate_freqdists(df, stop_dict):
    words = df['TotalEssayTokens'].tolist()
    words = [item for sublist in words for item in sublist]
    wordnet_lemmatizer = WordNetLemmatizer()
    words = [wordnet_lemmatizer.lemmatize(w.lower()) for w in words if w not in string.punctuation and w.lower() not in stop_dict]
    unigram_freq = nltk.FreqDist(words)
    
    bigrams = nltk.ngrams(words, 2)
    bigram_freq = nltk.FreqDist(bigrams)
    
    trigrams = nltk.ngrams(words, 3)
    trigram_freq = nltk.FreqDist(trigrams)
    
    return unigram_freq, bigram_freq, trigram_freq


# In[6]:

#Generates a vocabularly for the count vectorizer by computing the pmi for bigrams and trigrams like the paper
def generate_vocab(freq_dists):
    vocab = []
    unigram_freq = freq_dists[0]
    for k in unigram_freq.keys():
        if unigram_freq[k] >= 3:
            vocab.append(k)
    print(vocab[:10])
    
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
    print(tricount)
    for k in trigram_freq.keys():
        num = trigram_freq[k] / tricount
        denom = (unigram_freq[k[0]] / unicount) * (unigram_freq[k[1]] / unicount) * (unigram_freq[k[2]] / unicount)
        if num / denom > 6:
            vocab3.append(k)

    return set(vocab + vocab2 + vocab3)


# In[7]:

#Filters out words from the vocabulary v that are used by less than 1% of the users in the dataframe
def filter_vocab(df, v):
    #Every word gets a dictionary entry
    v_dict = {}
    for w in v:
        v_dict[w] = 0
    wordnet_lemmatizer = WordNetLemmatizer()
    
    #Going through every essay's unigrams, bigrams, trigrams
    #only adding to dictionary once if they exist in that essay
    essays = df['TotalEssayTokens'].tolist()
    for i,e in enumerate(essays):
        if i % 1000 == 0:
            print(str(i))
        words = [wordnet_lemmatizer.lemmatize(w.lower()) for w in e if w not in string.punctuation and w.lower() not in stop_dict]
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


# In[8]:

#Creates the data matrix, normalizes by user word count, writes the matrix to a pickle file just in case
#returns data matrix if needed.
def create_data_matrix(df, vocab, filename):
    
    count_vect = CountVectorizer(stop_words='english', tokenizer=happyfuntokenizing.Tokenizer().tokenize,
                                 ngram_range=(1, 3), analyzer='word', vocabulary=vocab)
    data_matrix = count_vect.fit_transform(df['TotalEssays'])
    
    data_matrix_dense = data_matrix.todense().astype(float)
    
    #Normalizing each row by the number of words that user uses
    essays = main_df['TotalEssayTokens'].tolist()
    for i,e in enumerate(essays):
        data_matrix_dense[i, :] = data_matrix_dense[i, :] / len(e)
        if i % 1000 == 0:
            print(str(i))
            
    with open(filename, 'wb') as f:
        pickle.dump(data_matrix_dense, f)
        
    return data_matrix
    


# In[9]:

main_df = get_data()


# In[10]:

main_df = tokenize_words(main_df)


# In[11]:

freq_dists = generate_freqdists(main_df, stop_dict)
print(freq_dists[0].most_common(10))
print()
print(freq_dists[1].most_common(10))
print()
print(freq_dists[2].most_common(10))


# In[12]:

vocab = generate_vocab(freq_dists)


# In[13]:

vocab = filter_vocab(main_df, vocab)


# In[14]:

data_matrix = create_data_matrix(main_df, vocab, 'datamatrix.pkl')


# In[15]:

np.isnan(data_matrix.todense()).sum()


# In[ ]:



