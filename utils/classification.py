from sklearn.cross_validation import train_test_split


def betas(clf, X, y, test_size=0.25):
    """The coefficients for a model fit using `clf`

    Parameters
    ----------
    clf
        Scikit-Learn classifier, e.g., `sklearn.linear_model.LogisticRegression`

    X : np.ndarray
        The features of dimension (n_samples, n_features)

    y : np.ndarray
        The 1d labels (targets) with length n_samples

    test_size : float between 0.0 and 1.0
        The decimal value for the proportion of test samples

    Returns
    -------
    beta_hats : np.ndarray
        Dependent on the `clf`. For logistic regression, `beta_hats` will be of
        dimension (n_classes, n_features); for random forest classification,
        `beta_hats` will be of length n_features.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=42)
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    if hasattr(clf, 'coef_'):
        beta_hats = clf.coef_
    elif hasattr(clf, 'feature_importances_'):
        beta_hats = clf.feature_importances_
    else:
        beta_hats = None
    return beta_hats
