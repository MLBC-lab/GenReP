import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def summarize_data(df):
    """
    Print basic summary statistics and info about the DataFrame.
    """
    print("DataFrame shape:", df.shape)
    print("DataFrame info:")
    print(df.info())
    print("\nDataFrame describe:")
    print(df.describe())

def check_missing_values(df):
    """
    Check for missing values in the DataFrame and print the result.
    """
    missing = df.isnull().sum()
    print("\nMissing values by column:")
    print(missing[missing > 0])

def find_correlations(df, method='pearson'):
    """
    Return a correlation matrix for numeric columns of the DataFrame.
    """
    return df.corr(method=method)

def distribution_by_group(df, numeric_col, group_col):
    """
    Compare the distribution of a numeric column across different groups.
    Creates a boxplot and a swarmplot for a more detailed view.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=group_col, y=numeric_col, data=df, palette='Set2')
    sns.swarmplot(x=group_col, y=numeric_col, data=df, color='k', alpha=0.5)
    plt.title(f"Distribution of {numeric_col} by {group_col}")
    plt.show()

def detect_outliers_zscore(df, columns, threshold=3.0):
    """
    Detect outliers using the z-score method for specified columns.
    Prints the number of outliers found in each column.
    """
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std(ddof=0))
            outliers = z_scores > threshold
            print(f"Column '{col}': {outliers.sum()} outliers detected.")
        else:
            print(f"Column '{col}' not found or not numeric.")

def advanced_eda_report(df, group_col=None):
    """
    Generate an advanced EDA report with group-based statistics and correlation.
    If group_col is provided, shows group-based mean and standard deviation 
    for numeric columns.
    """
    print("="*50)
    print("Basic Statistics")
    print("="*50)
    print(df.describe())
    print("\n")
    
    if group_col and group_col in df.columns:
        print("="*50)
        print(f"Group-Based Stats by '{group_col}'")
        print("="*50)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        group_stats = df.groupby(group_col)[numeric_cols].agg(['mean','std'])
        print(group_stats)
        print("\n")
    
    print("="*50)
    print("Correlation Heatmap")
    print("="*50)
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()
