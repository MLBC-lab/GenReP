import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def pca_reduction(X, n_components=2):
    """
    Perform PCA on feature matrix X to reduce dimensionality.
    Returns the transformed data and the PCA object.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def select_k_best(X, y, k=10, score_func=f_classif):
    """
    Select K best features using a specified scoring function.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)

def mutual_info_feature_selection(X, y, k=10):
    """
    Select K best features using mutual information.
    """
    selector = SelectKBest(mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)
