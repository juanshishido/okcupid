from collections import defaultdict
import re
from string import punctuation

import pandas as pd
from spacy.en import English

from utils.permutation import print_pvalues
from utils.text_representation import _levels, _multinomial


nlp = English(tagger=True, entity=False)

def tagger(doc):
    """For tagging a document
    Yields a (token, part-of-speech) tag tuple

    Parameters
    ----------
    doc : str
        A document with tokens to tag

    Yields
    ------
    tuple
        (token, tag)
    """
    text = nlp(doc)
    for sent in text.sents:
        for token in sent:
            yield (str(token), str(token.pos_))

def tag_corpus(corpus):
    """For tagging corpus document tokens

    Parameters
    ----------
    corpus : array-like
        A collection of documents

    Returns
    -------
    tagged : list
        (token, tag) tuples
    """
    assert isinstance(corpus, (list, pd.Series))
    tagged = []
    for doc in corpus:
        tagged.extend(tagger(doc))
    return tagged

def pos_tokens(tagged, pos):
    """Extract particular part-of-speech tokens

    Parameters
    ----------
    tagged : list
        (token, tag) tuples
    pos : str
        A valid part-of-speech tag

    Returns
    -------
    list

    Notes
    -----
    The available tags are:
        ADJ, ADP, ADV, AUX, CONJ, DET, INTJ, NOUN, NUM, PART,
        PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X, EOL, SPACE
    Source: https://spacy.io/docs#token-postags
    """
    return [t for t, p in tagged if p == pos]

def _pos_freq(doc):
    """Part of speech frequencies for individual documents
    
    Parameters
    -----------
    doc : str
        A document with tokens to tag
        
    Returns
    -------
    pos : dict
        With counts by tag
    """
    pos = defaultdict(float)
    for _, p in tagger(doc):
        pos[p] += 1
    return pos

def pos_df(corpus):
    """Create a DataFrame of part of speech
    frequencies for a corpus of documents
    
    Parameters
    ----------
    corpus : array-like
        A collection of documents
        
    Returns
    -------
    df : pd.DataFrame
    """
    assert isinstance(corpus, (list, pd.Series))
    pos_dfs = []
    for doc in corpus:
        frequencies = pd.DataFrame(_pos_freq(doc), index=[0])
        pos_dfs.append(frequencies)
    df = pd.concat(pos_dfs, ignore_index=True)
    df.fillna(0.0, inplace=True)
    return df

def pos_normalize(df):
    """Normalize (row-wise) part-of-speech frequencies

    Parameters
    ----------
    df : pd.DataFrame
        `pos_df()` DataFrame

    Returns
    -------
    pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    return (df.T / df.sum(axis=1)).T

def _arrs_pos(df_orig, df_pos, demographic, pos,
              d_levels=None, print_levels=False):
    """Individual part-of-speech
    arrays for a particular demographic
    
    Parameters
    ----------
    df_orig : pd.DataFrame
        The DataFrame from which `df_pos` was created
    df_pos : pd.DataFrame
        The part-of-speech DataFrame
    demographic : str
        A valid demographic-data column in `df_orig`
    pos : str
        A column in `df_pos` corresponding
        to a part of speech
    d_levels : list, default None
        The specific demographic levels desired
    print_levels : bool, default False
        Whether to print the demographic levels
    
    Returns
    -------
    arrs : tuple of np.arrays
        The corresponding `pos` values for each `demographic`
    """
    df_pos = df_pos.copy() # so we don't modify it
    df_pos[demographic] = df_orig[demographic].values
    levels = _levels(df_orig[demographic], d_levels, print_levels)
    arrs = []
    for d in levels:
        arr = df_pos[df_pos[demographic] == d][pos].values
        n = arr.shape[0]
        if n < 0.1 * df_pos.shape[0]:
            print("Warning: '" + d +
                  "' category has less than 10% of observations (" +
                  str(n) + ")")
        arrs.append(arr)
    return tuple(arrs)

def pos_by_split(df_orig, df_pos, demographic, pos=None,
                 d_levels=None, print_levels=False):
    """Wrapper for handling multiple parts-of-speech with `_arrs_pos()`

    Parameters
    ----------
    df_orig : pd.DataFrame
        The DataFrame from which `df_pos` was created
    df_pos : pd.DataFrame
        The part-of-speech DataFrame
    demographic : str
        A valid demographic-data column in `df_orig`
    pos : list, default None
        Parts-of-speech to compare
    d_levels : list, default None
        The specific demographic levels desired
    print_levels : bool, default False
        Whether to print the demographic levels

    Returns
    -------
    None

    Notes
    -----
    The number of unique values in `demographic` must be two
    """
    assert (isinstance(df_orig, pd.DataFrame) and
            isinstance(df_pos, pd.DataFrame))
    assert df_orig.shape[0] == df_pos.shape[0]
    assert demographic in df_orig.columns
    assert set(pos).issubset(df_pos.columns)
    for p in pos:
        a, b = _arrs_pos(df_orig, df_pos, demographic, p, d_levels, print_levels)
        print(p)
        print_pvalues(a, b)
        print()

def load_words(path):
    """To load profane and slang words

    Parameters
    ----------
    path : str
        Relative or absolute filepath

    Returns
    -------
    list
    """
    assert isinstance(path, str)
    with open(path, 'r') as f:
        return list(set([w.rstrip() for w in f.readlines()]))

def _contains_n(words, corpus):
    """Count the number of times a document contains particular words

    Parameters
    ----------
    words : list
        Words to check for
    corpus : array-like
        A collection of documents

    Returns
    -------
    np.ndarray
        Number of tokens by document
    """
    X, _ = _multinomial(corpus, {'vocabulary' : words})
    return X.toarray().sum(axis=1)

def contains(words, corpus):
    """Determine whether a document contains particular words

    Parameters
    ----------
    words : list
        Words to check for
    corpus : array-like
        A collection of documents

    Returns
    -------
    n_words : np.ndarray
        Binary representation
    """
    assert isinstance(words, list)
    assert isinstance(corpus, (list, pd.Series))
    n_words = _contains_n(words, corpus)
    n_words[n_words > 0] = 1
    return n_words
