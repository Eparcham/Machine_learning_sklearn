import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

def normalization(x):
    return (x-x.min())/(x.max()-x.min())

def standardize(x):
    return (x-x.mean())/x.std()

# ========= Main Function ========== #
def main():
    normz = MinMaxScaler(feature_range=(0, 1))
    standard = StandardScaler()
    print("Loading and preprocessing data...")
    dfe, df_raw = load_and_clean_data('./data/insurance.csv')

    print("\nEncoding categorical features...")
    df_encoded = label_encode_features(dfe)

    # df_encoded = df_encoded.apply(normalization)
    # print(df_encoded.head())
    df_encoded_array1 = normz.fit_transform(df_encoded)
    df_encoded_array2  = standard.fit_transform(df_encoded)








if __name__ == "__main__":
    main()


