import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from utils.lexical_features import pos_normalize
from utils.text_representation import _levels


def _levels_dict(demographics):
    """Create a dictionary of indices and
    corresponding demographic levels

    Parameters
    ----------
    demographics : pd.Series
        Demographic labels

    Returns
    -------
    levels_dict : dict
    """
    levels = _levels(demographics)
    levels_dict = {k : v for k, v in enumerate(levels)}
    return levels_dict

def most_like(demographics, doc, demo):
    """The demographic levels each user is most like

    Parameters
    ----------
    demographics : pd.Series
        Demographic labels
    doc : scipy.sparse.csr.csr_matrix
        Document-level tfidf matrix
    demo : scipy.sparse.csr.csr_matrix
        Demographic-level tfidf matrix

    Returns
    -------
    ct : pd.DataFrame
        Normalized (row-wise) demographic level counts
    """
    d_labels = _levels_dict(demographics)
    scores = cosine_similarity(doc, demo)
    df = pd.DataFrame(np.argmax(scores, axis=1), columns=['which'])
    df['most_like'] = df.which.apply(lambda x : d_labels[x])
    df['actual'] = demographics.values
    ct = pd.crosstab(df.actual, df.most_like)
    return pos_normalize(ct)
