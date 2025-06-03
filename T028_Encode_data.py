import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('./data/insurance.csv')

# Basic info and missing value overview
print("🔎 Dataset Information:")
print(df.info())
print("-" * 50)
print("📉 Missing Values Count (NaN):")
print(df.isna().sum())
print("-" * 50)

# Visualize missing data
msno.matrix(df)
plt.title("Missing Data Matrix", fontsize=16)
plt.show()

# ------------------------------------------
# Diagnose missing values column-wise
# ------------------------------------------
def diagnose_mv(df, mv_column):
    cols = df.columns
    flags = df[mv_column].isna()
    fig, ax = plt.subplots(len(cols), 3, figsize=(12, len(cols) * 3), constrained_layout=True)
    plt.rcParams['axes.grid'] = True

    for i, col in enumerate(cols):
        n1, bins, _ = ax[i, 0].hist(df[col].dropna(), bins=20)
        ax[i, 0].set_title(f"{col} (All)")

        n2, _, _ = ax[i, 1].hist(df.loc[~flags, col].dropna(), bins=bins)
        ax[i, 1].set_title(f"{col} (Non-missing {mv_column})")

        ax[i, 2].bar(bins[:-1], np.abs(n2 - n1), width=np.diff(bins))
        ax[i, 2].set_title(f"{col} (Diff)")
    plt.show()

# Example usage (disabled by default)
# diagnose_mv(df, 'bmi')

# ------------------------------------------
# Handle missing values
# ------------------------------------------

# 1. NCAR (e.g., 'children' likely Missing Completely At Random)
df['children'].fillna(df['children'].mode()[0], inplace=True)

# 2. MAR (e.g., 'bmi' depends on 'smoker')
median_bmi_smoker = df.loc[df['smoker'] == 'yes', 'bmi'].median()
df['bmi'].fillna(median_bmi_smoker, inplace=True)

# 3. NMAR (e.g., 'age' - drop if crucial)
df.dropna(subset=['age'], inplace=True)

# Confirm no missing values remain
msno.matrix(df)
plt.title("Missing Data Matrix After Cleaning", fontsize=16)
plt.show()

# ------------------------------------------
# Encoding categorical variables
# ------------------------------------------

# One-hot encoding for non-ordinal categorical variables
df_encoded = pd.get_dummies(df, drop_first=True, dtype=np.float64, columns=['sex', 'smoker', 'region'])
print("🔢 Encoded Dataset Preview:")
print(df_encoded.head(10))

# Optional: Label encoding (not used here as one-hot is preferred for non-ordinal)
label_encoded_df = df.copy()
label_encoder = LabelEncoder()
for col in label_encoded_df.select_dtypes(include='object'):
    label_encoded_df[col] = label_encoder.fit_transform(label_encoded_df[col])
    print(f"{col}: {label_encoder.classes_}")

print("\n🔤 Label Encoded Data Preview:")
print(label_encoded_df.head(10))
