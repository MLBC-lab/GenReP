import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_histogram(df, column, bins=30):
    """
    Plot a histogram of a specified DataFrame column.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, bins=bins, kde=True)
    plt.title(f"Histogram of {column}")
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot a heatmap of the correlation matrix.
    """
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col=None):
    """
    Create a scatter plot for two columns, with optional hue.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
    plt.title(f"Scatter Plot of {x_col} vs {y_col}")
    plt.show()

def pca_scatter(X, labels=None):
    """
    Create a scatter plot after reducing the dimensionality of X (2D) using PCA externally.
    X is assumed to be already projected into 2D. labels can be used to color points.
    """
    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette='Set2')
    else:
        plt.scatter(X[:,0], X[:,1], alpha=0.7, edgecolor='k')
    plt.title("PCA 2D Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
