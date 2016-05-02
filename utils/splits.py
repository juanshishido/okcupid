import numpy as np
import pandas as pd


def counts_by_class(cm, df, col, one_vs_one=True, vals=None):
    """Aggregates token frequencies for a particular binary
    split. The split is based on an input DataFrame, whose
    rows must correspond to those in `cm`. This supports
    both one-vs-one and one-vs-rest splits.

    Parameters
    ----------
    cm : scipy.sparse.csr.csr_matrix
        the `count_matrix` from `col_to_data_matrix()`
    df : pd.DataFrame
        the DataFrame that `cm` is based off of
    col : str
        valid column name in `df`
    one_vs_one : bool
        whether the split is based on two distinct values
    vals : list, int, str
        the value to split on, whose type depends on
        both `col` and `one_vs_one`

    Returns
    -------
    counts : np.ndarray
        shape (2, n_features)
    """
    if one_vs_one:
        assert isinstance(vals, list) and len(vals) == 2
        c0 = cm[np.array(df[col]==vals[0]), :].sum(axis=0)
        c1 = cm[np.array(df[col]==vals[1]), :].sum(axis=0)
    else:
        assert isinstance(vals, (int, str))
        c0 = cm[np.array(df[col]==vals), :].sum(axis=0)
        c1 = cm[np.array(df[col]!=vals), :].sum(axis=0)
    counts = np.array(np.vstack((c0, c1)))
    return counts

def diff_prop(counts, vocabulary):
    """Calculate the difference in relative token frequency

    Parameters
    ----------
    counts : np.ndarray
        shape (2, n_features)
    vocabulary : list
        the vocabulary list from `col_to_data_matrix()`

    Returns
    -------
    df : pd.DataFrame
        with single column corresponding to the difference in
        proportions metric and with tokens are indices
    """
    proportions = counts / counts.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(proportions.T)
    df['diff'] = df[0] - df[1]
    df.index = vocabulary
    return df[['diff']]

def wf(df, n):
    """Create the list of (token, value) tuples for `wcloud`

    Parameters
    ----------
    df : pd.DataFrame
        assuming single column corresponding to either `diff`
        or `log_odds_ratio`
    n : int
        number of terms to capture

    Returns
    -------
    wf_top, wf_bottom : list, list
        (token, value) tuples
    """
    df = df.copy()
    col = df.columns[0]
    tmp = df.sort(col, ascending=False)
    top = tmp.iloc[:n]
    bottom = tmp.iloc[-n:]
    wf_top = list(zip(top.index, top[col]))
    wf_bottom = list(zip(bottom.index, bottom[col] * -1))
    return wf_top, wf_bottom

def subset_df(df, col, vals):
    """Return a subset of `df` based on particular `vals` for `col`

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    col : str
        Valid column name
    vals : list
        Values to subset on

    Returns
    -------
    subset : pd.DataFrame
        The rows in `df` with values in `val` for `col`
    """
    df = df.copy()
    subset = df[df[col].isin(vals)]
    return subset
