import numpy as np
from sklearn.metrics import accuracy_score


def _diff_means(m, arr):
    """Calculate the difference-in-means statistic.
    This is based on an input array, `arr`, where the first
    `m` observations correspond to a particular class.

    Parameters
    ----------
    m : int
        Number of samples in the first class
    arr : np.ndarray
        Data for both classes

    Returns
    -------
    float
    """
    return np.mean(arr[:m]) - np.mean(arr[m:])

def permute(a, b, comparison='predictions', permutations=10000):
    """Estimate of the permutation-based p-value

    Parameters
    ----------
    a : np.ndarray
        Data for one class or
        ground truth (correct) labels
    b : np.ndarray
        Data for another class or
        predicted labels, as returned by a classifier
    comparison : str
        {'predictions', 'means'}
    permutations : int, optional
        Number of permutations

    Returns
    -------
    p_value : float
        The proportion of times a value as extreme
        as the observed estimate is seen

    Notes
    -----
    This calculates the two-tailed p-value
    """
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    assert comparison in ('predictions', 'means')
    np.random.seed(42)
    if comparison == 'predictions':
        c = b.copy()
        compare = accuracy_score
    else:
        c = np.append(a, b)
        a = a.shape[0]
        compare = _diff_means
    baseline = compare(a, c)
    v = []
    for _ in range(permutations):
        np.random.shuffle(c)
        v.append(compare(a, c))
    p_value = (np.array(v) >= baseline).sum() / permutations
    return p_value
