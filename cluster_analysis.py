import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

def kmeans_clustering(X, n_clusters=3):
    """
    Perform K-Means clustering on feature matrix X.
    Returns the cluster labels, fitted model, and silhouette score.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    return labels, model, sil_score

def hierarchical_clustering(X, n_clusters=3, linkage='ward'):
    """
    Perform Agglomerative (hierarchical) clustering on feature matrix X.
    Returns the cluster labels, fitted model, and silhouette score.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    return labels, model, sil_score

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on feature matrix X.
    Returns the cluster labels, fitted model, and silhouette score (if valid).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    
    # If there's only one cluster or everything is outlier, skip silhouette
    unique_labels = set(labels)
    if len(unique_labels) > 1 and -1 not in unique_labels:
        sil_score = silhouette_score(X, labels)
    else:
        sil_score = None
    return labels, model, sil_score
