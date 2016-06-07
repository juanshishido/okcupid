import numpy as np
from sklearn.metrics import accuracy_score


def permute(a, b, permutations=10000):
    """Estimate of the permutation-based p-value

    Parameters
    ----------
    a : np.ndarray
        Data for one class or
        ground truth (correct) labels
    b : np.ndarray
        Data for another class or
        predicted labels, as returned by a classifier
    permutations : int, optional
        Number of permutations

    Returns
    -------
    p_value : float
        The proportion of times a value as extreme
        as the observed estimate is seen
    """
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    np.random.seed(42)
    b = b.copy()
    baseline = accuracy_score(a, b)
    v = []
    for _ in range(permutations):
        np.random.shuffle(b)
        v.append(accuracy_score(a, b))
    p_value = (np.array(v) >= baseline).sum() / permutations
    return p_value
