import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize


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
    """Use regular expression tokenizer.
    Keep apostrophes.
    Returns a list of lists, one list for each sentence:
        [[word, word], [word, word, ..., word], ...].
    """
    if type(text) is list:
        return [regexp_tokenize(sentence, pattern="\w+(?:[-']\w+)*")
                for sentence in text]
    else:
        return regexp_tokenize(text, pattern="\w+(?:[-']\w+)*")

def normalize(tokenized_words, remove_stopwords = True):
    """Removes stop words, numbers, short words, and lowercases text.
    Returns a list of lists, one list for each sentence:
        [[word, word], [word, word, ..., word], ...].
    """
    if remove_stopwords:
        stop_words = stopwords.words('english')
        return [[w.lower() for w in sent
                 if (w.lower() not in stop_words) and\
                 (not(w.lower().isnumeric())) and\
                 (len(w) > 2)]
                for sent in tokenized_words]
    else:
        return [[w.lower() for w in sent
                if not(w.lower().isnumeric())]
                for sent in tokenized_words]
