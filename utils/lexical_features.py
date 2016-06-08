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

def pos_df(corpus, normalize=True):
    """Create a DataFrame of part of speech
    frequencies for a corpus of documents
    
    Parameters
    ----------
    corpus : array-like
        A collection of documents
    normalize : bool (default True)
        Whether to normalize by token counts
        
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
    if normalize:
        df = (df.T / df.sum(axis=1)).T
    return df

def _levels(df, demo, d_levels=None):
    """The demographic levels to iterate over
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame with the demographic data
    
    demo : str
        A valid demographic-data column in `df`
    
    d_levels : list of None (default)
        The specific demographic levels desired
    
    Returns
    -------
    levels : list
        The unique (sorted) levels in `df[demo]`
    """
    levels = df[demo].unique()
    if d_levels:
        assert set(d_levels).issubset(levels)
        levels = d_levels
    levels.sort()
    print('Levels (in order):', levels, end='\n\n')
    return levels

def arrs_pos(df_orig, df_pos, demographic, pos, d_levels=None):
    """Individual part-of-speech
    arrays for a particular demographic
    
    Parameters
    ----------
    df_orig : pd.DataFrame
        The DataFrame from which `df_pos` was created
    
    pos_df : pd.DataFrame
        The part-of-speech DataFrame
    
    demographic : str
        A valid demographic-data column in `df_orig`
    
    pos : str
        A column in `df_pos` corresponding
        to a part of speech
    
    d_levels : list or None (default)
        The specific demographic levels desired
    
    Returns
    -------
    arrs : tuple of np.arrays
        The corresponding `pos` values for each `demographic`
    """
    assert (isinstance(df_orig, pd.DataFrame) and
            isinstance(df_pos, pd.DataFrame))
    assert df_orig.shape[0] == df_pos.shape[0]
    assert (demographic in df_orig.columns and
            pos in df_pos.columns)
    df_pos = df_pos.copy() # so we don't modify it
    df_pos[demographic] = df_orig[demographic].values
    levels = _levels(df_orig, demographic, d_levels)
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
