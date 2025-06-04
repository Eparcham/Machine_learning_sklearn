import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.ops.gen_array_ops import upper_bound
from traitlets.utils.descriptions import describe


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


# ========= Main Function ========== #
def main():
    print("Loading and preprocessing data...")
    dfe, df_raw = load_and_clean_data('./data/insurance.csv')

    print("\nEncoding categorical features...")
    df_encoded = label_encode_features(dfe)

    fig,axes = plt.subplots(1,2,figsize=(12,6))
    axes[0].boxplot(df_encoded['bmi'])
    axes[0].violinplot(df_encoded['bmi'])
    axes[1].scatter(df_encoded['bmi'], df_encoded['charges'])



    describe = df_encoded.describe()
    q1 = describe.loc['25%','bmi']
    q2 = describe.loc['50%','bmi']
    q3 = describe.loc['75%','bmi']


    iqr = q3-q1
    lower_cap = q1-1.5* iqr
    upper_cap = q3+1.5* iqr

    print(q1, q2, q3, iqr, lower_cap, upper_cap)
    flags = (df_encoded.bmi<lower_cap)| (df_encoded.bmi>upper_cap)
    axes[1].scatter(df_encoded['bmi'][flags], df_encoded['charges'][flags])
    plt.show()

    df_encoded.bmi[flags] = upper_cap

    ## Bivariate analysis
    fig,axes = plt.subplots(3,3,figsize=(12,12))
    axes_flat = axes.flatten()

    for i , col in enumerate(df_encoded.columns):
        ax=axes_flat[i]
        ax.scatter(df_encoded[col], df_encoded.charges)
        flags = df_encoded.charges >50000
        ax.scatter(df_encoded[col][flags], df_encoded.charges[flags])
        ax.set_title(col)

    plt.show()

    index = df_encoded[flags].index
    df_encoded.drop(index, inplace=True)






if __name__ == "__main__":
    main()
