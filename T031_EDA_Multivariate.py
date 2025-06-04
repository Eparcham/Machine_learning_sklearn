import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder

# ========== Data Loading and Cleaning ========== #
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

    # Show missing data matrix
    print("Missing values visualization:")
    msno.matrix(df)
    plt.show()

    # Handle missing values
    dfe['children'].fillna(df['children'].mode()[0], inplace=True)
    median_bmi_smoker = dfe.loc[dfe['smoker'] == 'yes', 'bmi'].median()
    dfe['bmi'].fillna(median_bmi_smoker, inplace=True)
    dfe.dropna(subset=['age'], inplace=True)

    return dfe, df

# ========== Encoding ========== #
def label_encode_features(df):
    """
    Label encode all categorical columns and print their mappings.
    """
    df_encoded = df.copy()
    encoder = LabelEncoder()

    print("\nLabel Encoding Mappings:")
    for col in df_encoded.select_dtypes(include='object'):
        df_encoded[col] = encoder.fit_transform(df_encoded[col])
        print(f" - {col}: {list(encoder.classes_)}")

    return df_encoded

# ========== Visualization Helpers ========== #
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
    """Show scatter and pairplot for exploratory analysis."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df_encoded['age'], df_encoded['charges'], alpha=0.6, c='purple')
    plt.title("Age vs Charges")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.grid(True)
    plt.show()

    print("Generating pairplot...")
    sns.pairplot(df_raw)
    plt.suptitle("Pairplot of Raw Data", fontsize=16)
    plt.show()

def categorical_visuals(df):
    """Visualize relationship between categorical and numeric data."""
    print("\nCrosstab: Sex vs Number of Children")
    print(pd.crosstab(index=df['sex'], columns=df['children']))

    plt.figure(figsize=(10, 5))
    sns.boxplot(x='sex', y='charges', data=df)
    plt.title("Boxplot of Charges by Sex")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.violinplot(x='sex', y='charges', data=df)
    plt.title("Violin Plot of Charges by Sex")
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.violinplot(x='sex', y='charges', data=df, inner=None, color="skyblue")
    sns.boxplot(x='sex', y='charges', data=df, width=0.2, boxprops={'zorder': 2})
    plt.title("Combined Violin and Boxplot")
    plt.show()

# ========== Multi-variable Visuals ========== #
def multivariable_visuals(df_encoded):
    """Visualize interactions between 3+ variables (e.g., smoker, BMI, age, charges)."""
    print("\nScatter: Smoker vs BMI vs Charges")
    flags = df_encoded['smoker'] == 1

    # Smoker's BMI vs Charges
    plt.figure(figsize=(8, 6))
    plt.scatter(df_encoded.bmi[flags], df_encoded.charges[flags], c='darkred', alpha=0.6)
    plt.title("Charges vs BMI (Smokers Only)")
    plt.xlabel("BMI")
    plt.ylabel("Charges")
    plt.grid(True)
    plt.show()

    # Smoker's Age vs Charges
    plt.figure(figsize=(8, 6))
    plt.scatter(df_encoded.age[flags], df_encoded.charges[flags], c='navy', alpha=0.6)
    plt.title("Charges vs Age (Smokers Only)")
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.grid(True)
    plt.show()

    # 3D-style plot using hue
    print("Plotting: Charges vs BMI vs Smoker")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df_encoded, palette='Set1')
    plt.title("Charges vs BMI by Smoking Status")
    plt.show()

# ========== Main Function ========== #
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

    print("\nGenerating multi-variable visualizations...")
    multivariable_visuals(df_encoded)


if __name__ == "__main__":
    main()
