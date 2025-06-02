import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

# Load the dataset
df = pd.read_csv('./data/insurance.csv')

# Show info and missing values
print("🔎 Dataset Information:")
print(df.info())
print("-" * 50)
print("📉 Missing Values Count (NaN):")
print(df.isna().sum())
print("-" * 50)

# Plot missing values visually
msno.matrix(df)
plt.title("Missing Data Matrix", fontsize=16)
plt.show()

# ------------------------------------------
# Automatically diagnose all missing features
# ------------------------------------------

def diagnose_mv(df, mv_column):
    cols = df.columns
    flags = df[mv_column].isna()
    fig,ax = plt.subplots(len(cols),3,figsize=(len(cols)+3,len(cols)+3),constrained_layout=True)
    plt.rcParams['axes.grid'] = True

    for i,col in enumerate(cols):
        n1, bins,_ = ax[i,0].hist(df[col])
        ax[i,0].set_title(f"{col} with mv")

        n2, bins,_ = ax[i,1].hist(df[col][~flags],bins=bins)
        ax[i,1].set_title(f"{col} without mv")

        if col == "charges":
            bins/=1e4

        ax[i,2].bar(bins[:-1], np.abs(n2-n1))
        ax[i,2].set_title(f"{col} Difference")
    plt.show()

if 0:
    diagnose_mv(df, 'age')

    diagnose_mv(df, 'bmi')


## resolve missing value

# NCAR
mode_children = df['children'].mode().values
print(mode_children)


df['children'] = df['children'].fillna(mode_children[0])

print(df['children'].isna().sum())

## MAR
flags = df.smoker=='yes'
print(flags.sum())
m = df.bmi[flags].median()
# df.bmi.fillna(m, inplace=True)
df['bmi'] = df['bmi'].fillna(m)
msno.matrix(df)
plt.title("Missing Data Matrix", fontsize=16)
plt.show()

## NMAR

flags = df['age'].isna()
df.dropna(subset=['age'], inplace=True)
msno.matrix(df)
plt.title("Missing Data Matrix", fontsize=16)
plt.show()
