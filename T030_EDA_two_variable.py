import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(file_path):
    """
    Load and clean the insurance dataset.

    Handling missing values:
    - MCAR: Fill 'children' with mode.
    - MAR: Fill 'bmi' with median BMI of smokers.
    - NMAR: Drop rows with missing 'age'.
    """
    df = pd.read_csv(file_path)
    dfe = df.copy()

    # Visualize missing data
    print("Missing values visualization:")
    msno.matrix(df)
    plt.show()

    # Handle missing values
    dfe['children'].fillna(df['children'].mode()[0], inplace=True)
    median_bmi_smoker = dfe.loc[dfe['smoker'] == 'yes', 'bmi'].median()
    dfe['bmi'].fillna(median_bmi_smoker, inplace=True)
    dfe.dropna(subset=['age'], inplace=True)

    return dfe, df


def label_encode_features(df):
    """
    Label encode categorical features and print label mappings.
    """
    df_encoded = df.copy()
    encoder = LabelEncoder()

    print("\nLabel Encoding Mappings:")
    for col in df_encoded.select_dtypes(include='object'):
        df_encoded[col] = encoder.fit_transform(df_encoded[col])
        print(f" - {col}: {list(encoder.classes_)}")

    return df_encoded


def plot_histograms(df):
    """Plot histograms for all numeric features."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols].hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
    plt.suptitle("Histograms of Numeric Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_boxplots(df):
    """Plot boxplots for all numeric features."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[col], color='lightgreen')
        plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()


def correlation_analysis(df):
    """Plot correlation heatmap of numeric features."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap", fontsize=14)
    plt.show()


def scatter_and_pairplot(df_encoded, df_raw):
    """Plot scatter and pairplot."""
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_encoded['age'], df_encoded['charges'], alpha=0.6, c='purple')
    plt.title("Age vs Charges")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.grid(True)
    plt.show()

    # Pairplot
    print("Generating pairplot...")
    sns.pairplot(df_raw)
    plt.suptitle("Pairplot of Raw Data", fontsize=16)
    plt.show()


def categorical_visuals(df):
    """Show boxplots, violin plots, and crosstabs by gender and children."""
    print("\nCrosstab: Sex vs Number of Children")
    print(pd.crosstab(index=df['sex'], columns=df['children']))

    # Boxplot by sex
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='sex', y='charges', data=df)
    plt.title("Boxplot of Charges by Sex")
    plt.show()

    # Violin plot by sex
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='sex', y='charges', data=df)
    plt.title("Violin Plot of Charges by Sex")
    plt.show()

    # Combined box + violin
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='sex', y='charges', data=df, inner=None, color="skyblue")
    sns.boxplot(x='sex', y='charges', data=df, width=0.2, boxprops={'zorder': 2})
    plt.title("Combined Violin and Boxplot")
    plt.show()


def main():
    print("Loading and preprocessing data...")
    dfe, df_raw = load_and_clean_data('./data/insurance.csv')

    print("\nEncoding categorical features...")
    df_encoded = label_encode_features(dfe)

    print("\nPlotting histograms...")
    plot_histograms(dfe)

    print("\nPlotting boxplots...")
    plot_boxplots(dfe)

    print("\nAnalyzing correlations...")
    correlation_analysis(df_encoded)

    print("\nGenerating scatter and pairplot...")
    scatter_and_pairplot(df_encoded, df_raw)

    print("\nGenerating categorical visualizations...")
    categorical_visuals(dfe)

if __name__ == "__main__":
    main()
