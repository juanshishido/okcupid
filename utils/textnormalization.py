import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize

from .happyfuntokenizing import Tokenizer


def split_on_sentence(text):
    """Tokenize the text on sentences.
    Returns a list of strings (sentences).
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return sent_tokenizer.tokenize(text)

def re_punc(text):
    """Remove all punctuation. Keep apostrophes."""
    return re.sub(r'[!"#$%&()*+,\-\./:;<=>?@\[\]^_`\\{\|}]+', ' ', text)

def remove_punctuation(sentences):
    """Remove punctuation based on `re_punc`.
    Returns either a list of string or a single string,
    based on the input type.
    """
    if type(sentences) is list:
        return [re_punc(sentence).strip() for sentence in sentences]
    else:
        return re_punc(sentences).strip()

def split_on_word(text):
    """Use happyfuntokenizing tokenizer.
    Returns a list of lists, one list for each sentence:
        [[word, word], [word, word, ..., word], ...].
    """
    hft = Tokenizer()
    if type(text) is list:
        return [hft.tokenize(sentence)
                for sentence in text]
    else:
        return hft.tokenize(text)

def normalize(tokenized_words):
    """Removes stop words and lowercases text.
    Returns a list of lists, one list for each sentence:
        [[word, word], [word, word, ..., word], ...].
    """
    stop_words = stopwords.words('english')
    return [[w.lower() for w in sent
             if (w.lower() not in stop_words)]
            for sent in tokenized_words]
