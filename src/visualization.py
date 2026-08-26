"""
visualization.py

Contains reusable visualization functions for Exploratory Data Analysis (EDA).

Author: Anchit
Project: Customer Churn Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Global Plot Style
# -------------------------------------------------------------------

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# -------------------------------------------------------------------
# 1. Histogram with KDE
# -------------------------------------------------------------------

def plot_histogram(df, column, bins=30):
    """
    Plot histogram with KDE for a numerical feature.
    """
    plt.figure()

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 2. Box Plot
# -------------------------------------------------------------------

def plot_boxplot(df, column):
    """
    Plot boxplot for a numerical feature.
    """
    plt.figure()

    sns.boxplot(
        data=df,
        y=column
    )

    plt.title(f"Boxplot of {column}")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 3. Count Plot
# -------------------------------------------------------------------

def plot_countplot(df, column):
    """
    Plot countplot for a categorical feature.
    """
    plt.figure()

    sns.countplot(
        data=df,
        x=column
    )

    plt.title(f"Countplot of {column}")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 4. Count Plot with Target
# -------------------------------------------------------------------

def plot_countplot_hue(df, column, target):
    """
    Plot categorical feature against target.
    """
    plt.figure(figsize=(7,5))

    sns.countplot(
        data=df,
        x=column,
        hue=target
    )

    plt.title(f"{column} vs {target}")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 5. Box Plot with Target
# -------------------------------------------------------------------

def plot_boxplot_hue(df, feature, target):
    """
    Plot numerical feature against target.
    """
    plt.figure(figsize=(7,5))

    sns.boxplot(
        data=df,
        x=target,
        y=feature
    )

    plt.title(f"{feature} vs {target}")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 6. Scatter Plot
# -------------------------------------------------------------------

def plot_scatter(df, x, y):
    """
    Scatter plot for numerical vs numerical.
    """
    plt.figure(figsize=(6,5))

    sns.scatterplot(
        data=df,
        x=x,
        y=y
    )

    plt.title(f"{x} vs {y}")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 7. Correlation Heatmap
# -------------------------------------------------------------------

def plot_heatmap(df):
    """
    Plot correlation heatmap.
    """
    plt.figure(figsize=(10,8))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 8. Target Correlation Plot
# -------------------------------------------------------------------

def plot_target_correlation(df, target):
    """
    Plot correlation of every feature with target.
    """
    corr = (
        df.corr(numeric_only=True)[target]
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8,6))

    sns.barplot(
        x=corr.values,
        y=corr.index
    )

    plt.title(f"Feature Correlation with {target}")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 9. Pair Plot
# -------------------------------------------------------------------

def plot_pairplot(df, columns, hue=None):
    """
    Pairplot of selected columns.
    """
    sns.pairplot(
        data=df,
        vars=columns,
        hue=hue
    )

    plt.show()


# -------------------------------------------------------------------
# 10. Missing Values Plot
# -------------------------------------------------------------------

def plot_missing_values(df):
    """
    Plot percentage of missing values.
    """
    missing = (
        df.isnull()
          .sum()
          .sort_values(ascending=False)
    )

    missing = missing[missing > 0]

    plt.figure(figsize=(10,5))

    sns.barplot(
        x=missing.values,
        y=missing.index
    )

    plt.title("Missing Values")

    plt.xlabel("Count")

    plt.tight_layout()
    plt.show()
