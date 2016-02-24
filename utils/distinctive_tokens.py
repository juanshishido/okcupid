import numpy as np
import pandas as pd


def log_odds_ratio(counts, feature_names=None, symmetric_alpha=1, use_variance=False):
    """
    Parameters
    ----------
    counts : np.ndarray
        2d frequency counts with
        dimensions (`n_classes`, `n_features`)
    feature_names : list
        the token names with length `n_features`
    symmetric_alpha : int
        constant
    use_variance : bool
        whether to account for variance
    
    Returns
    -------
    log_odds : pd.DataFrame
        with feature names (one for each of the `n_features`)
        and log odds ratio values as distinct columns
    
    Notes
    -----    
    Is this equation 16 in Monroe et al.?
    
    If so, it looks like the `F` should be multiplied by
    `symmetric_alpha` in the denominator in `odds1`
    """
    assert isinstance(counts, np.ndarray), 'Must be type np.ndarray'
    assert counts.shape[1] == len(feature_names), ('the length of `feature_names`' +
                                                   'should equal the number of features in `counts`')
    F = counts.shape[1]
    sum0, sum1 = np.sum(counts, axis=1)
    odds0 = (np.log((counts[0] + symmetric_alpha) /
                    (sum0 + F * symmetric_alpha - counts[0] - symmetric_alpha)))
    odds1 = (np.log((counts[1] + symmetric_alpha) /
                    (sum1 + F - counts[1] - symmetric_alpha)))
    diff = odds0 - odds1
    if use_variance:
        diff = diff / np.sqrt((1 / (counts[0] + symmetric_alpha)) +
                              (1 / (counts[1] + symmetric_alpha)))
    log_odds = pd.DataFrame(diff, columns=['log_odds_ratio'])
    log_odds['features'] = feature_names
    return log_odds
