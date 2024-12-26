import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, pearsonr, spearmanr
from sklearn.metrics import confusion_matrix

def two_sample_ttest(sample1, sample2):
    """
    Perform a two-sample t-test between two arrays of data.
    Returns the t-statistic and p-value.
    """
    t_stat, p_value = ttest_ind(sample1, sample2, equal_var=False)
    return t_stat, p_value

def anova_test(*samples):
    """
    Perform a one-way ANOVA test on multiple samples.
    Returns the F-statistic and p-value.
    """
    f_stat, p_value = f_oneway(*samples)
    return f_stat, p_value

def correlation(x, y, method="pearson"):
    """
    Calculate correlation between two arrays using Pearson or Spearman method.
    Returns the correlation coefficient and p-value.
    """
    if method == "pearson":
        return pearsonr(x, y)
    elif method == "spearman":
        return spearmanr(x, y)
    else:
        raise ValueError("Invalid correlation method. Choose 'pearson' or 'spearman'.")

def classification_stats(y_true, y_pred):
    """
    Calculate classification metrics (accuracy, precision, recall, specificity, F1 score) 
    from the confusion matrix for a binary classification.
    Returns a dictionary of metrics.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # Suppose 0 = negative, 1 = positive
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-15)
    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)  # sensitivity
    specificity = tn / (tn + fp + 1e-15)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1_score": f1_score
    }
    return metrics

def multiclass_stats(y_true, y_pred, labels):
    """
    Compute confusion matrix and per-class precision, recall, and F1 
    for a multiclass classification scenario.
    Returns a dictionary of metrics for each class.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = {}
    
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)
        
        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    return results
