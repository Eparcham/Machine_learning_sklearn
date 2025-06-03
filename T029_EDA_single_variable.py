import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(file_path):
    """Load and preprocess the insurance dataset."""
    df = pd.read_csv(file_path)

    # Handle missing values
    # 1. MCAR: 'children' - replace with mode
    df['children'].fillna(df['children'].mode()[0], inplace=True)

    # 2. MAR: 'bmi' - fill based on 'smoker' median
    median_bmi_smoker = df.loc[df['smoker'] == 'yes', 'bmi'].median()
    df['bmi'].fillna(median_bmi_smoker, inplace=True)

    # 3. NMAR: 'age' - drop rows with missing age
    df.dropna(subset=['age'], inplace=True)

    return df


def encode_features(df):
    """Perform one-hot encoding for categorical variables."""
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=np.float64, columns=['sex', 'smoker', 'region'])
    return df_encoded


def label_encode_features(df):
    """Label encode categorical features and display label mappings."""
    df_label = df.copy()
    encoder = LabelEncoder()

    for col in df_label.select_dtypes(include='object'):
        df_label[col] = encoder.fit_transform(df_label[col])
        print(f"{col}: {encoder.classes_}")

    return df_label


def calculate_statistics(df):
    """Display central tendency and variability statistics."""
    print("\n=== Central Tendency ===")
    print("Mean:\n", df.mean())
    print("Median:\n", df.median())
    print("Mode:\n", df.mode().iloc[0])

    print("\n=== Variability ===")
    print("Standard Deviation:\n", df.std())
    print("Variance:\n", df.var())

    def median_abs_dev(x):
        return (x - x.median()).abs().median()

    print("\n=== Median Absolute Deviation ===")
    print("Full DataFrame:\n", df.apply(median_abs_dev))
    print("Age Column:\n", median_abs_dev(df["age"]))


def plot_histograms(df):
    """Plot histograms of all features."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_plots = len(numeric_cols)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes_flat = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes_flat[i].hist(df[col], bins=20)
        axes_flat[i].set_title(f"Histogram of {col}")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_boxplots(df):
    """Plot boxplots of all numeric features."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_plots = len(numeric_cols)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes_flat = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes_flat[i].boxplot(df[col])
        axes_flat[i].set_title(f"Boxplot of {col}")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # Load and process data
    df_raw = load_and_clean_data('./data/insurance.csv')

    # One-hot encoded DataFrame (for ML)
    df_encoded = encode_features(df_raw)

    # Label-encoded version (for statistics)
    df_labeled = label_encode_features(df_raw)

    # Statistics
    calculate_statistics(df_labeled)

    # Visualization
    plot_histograms(df_labeled)
    plot_boxplots(df_labeled)


if __name__ == "__main__":
    main()
