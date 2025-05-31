import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('./data/insurance.csv')

# Display the first and last 10 records
print("First 10 records:\n", df.head(10), "\n")
print("Last 10 records:\n", df.tail(10), "\n")

# Display data types and shape
print("Data types:\n", df.dtypes, "\n")
print("Dataset shape (rows, columns):", df.shape, "\n")

# Access specific rows and columns
print("First row data:\n", df.iloc[0], "\n")
print("Rows 0 to 2, columns 2 to 3:\n", df.iloc[0:3, 2:4], "\n")
print("Rows 0 and 20:\n", df.iloc[[0, 20]], "\n")

# Separator for clarity
print("-" * 60)

# Display specific columns
print("Column 'age':\n", df['age'], "\n")
print("-" * 60)

print("Columns 'age' and 'children':\n", df[['age', 'children']], "\n")
print("-" * 60)

# Alternative way to access the 'children' column
print("Column 'children' (dot notation):\n", df.children, "\n")
print("-" * 60)

# Display column names
print("Column names:\n", df.columns.tolist(), "\n")
print("-" * 60)

# Dataset info before category conversion
print("Dataset info before conversion:\n")
df.info()
print("-" * 60)

# Convert object-type columns to categorical
categorical_cols = ['sex', 'smoker', 'region']
df[categorical_cols] = df[categorical_cols].astype('category')

# Dataset info after conversion
print("Dataset info after converting to categorical:\n")
df.info()
