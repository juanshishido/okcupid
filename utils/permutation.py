import numpy as np
from sklearn.metrics import accuracy_score


def permute(y_true, y_pred, permutations=10000):
    """Permutation-based p-value for assessing classifier performance

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (correct) labels
    y_pred : np.ndarray
        Predicted labels, as returned by a classifier
    permutations : int, optional
        Number of times to permute `y_pred`

    Returns
    -------
    p_value : float
        Permutation-based

    Notes
    -----
    Using Scikit-Learn's `y_true` and `y_pred` naming conventions
    and descriptions

    The p-value represents the fraction of times where the accuracy
    of the shuffled `y_pred` labels and `y_true` labels was greater
    than or equal to the baseline accuracy, as predicted by the classifier
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    np.random.seed(42)
    y_pred = y_pred.copy()
    baseline = accuracy_score(y_true, y_pred)
    v = []
    for _ in range(permutations):
        np.random.shuffle(y_pred)
        v.append(accuracy_score(y_true, y_pred))
    p_value = (np.array(v) >= baseline).sum() / permutations
    return p_value
