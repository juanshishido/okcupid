from collections import defaultdict

import pandas as pd
from spacy.en import English


nlp = English(tagger=True, entity=False)

def _pos_freq(doc):
    """Part of speech frequencies for individual documents
    
    Parameters
    -----------
    doc : str
        A document with multiple sentences
        
    Returns
    -------
    pos : dict
    """
    text = nlp(doc)
    pos = defaultdict(float)
    for sent in text.sents:
        for token in sent:
            pos[token.pos_] += 1
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
