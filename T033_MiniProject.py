import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox, skew
from scipy.special import boxcox as boxcox_transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression,SGDRegressor
from scipy.stats import boxcox       # for fitting on train
from scipy.special import boxcox as boxcox_apply  # for applying λ to test


# ==========================
# 1. Data Loading & Overview
# ==========================
print("="*25 + " DATA LOADING " + "="*25)
df = pd.read_csv("./data/housing.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(f"Shape: {df.shape}")

# Convert categorical column
df['ocean_proximity'] = df['ocean_proximity'].astype('category')

# ==========================
# 2. Train-Test Split
# ==========================
print("\n" + "="*25 + " TRAIN-TEST SPLIT " + "="*25)
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train shape: {train_set.shape} | Test shape: {test_set.shape}")

# ==========================
# 3. Missing Value Handling
# ==========================
print("\n" + "="*25 + " MISSING VALUE VISUALIZATION " + "="*25)
msno.matrix(train_set)
plt.title("Missing Values in Training Set")
plt.show()

# Impute missing values for all float columns with median (applies to both sets)
float_cols = train_set.select_dtypes(include='float64').columns
imputer = SimpleImputer(strategy='median')
train_set[float_cols] = imputer.fit_transform(train_set[float_cols])
test_set[float_cols] = imputer.transform(test_set[float_cols])

# ==========================
# 4. Categorical Encoding
# ==========================
print("\n" + "="*25 + " ENCODING CATEGORICAL FEATURES " + "="*25)
print("Unique values:", train_set.ocean_proximity.unique())
train_set = pd.get_dummies(train_set, columns=['ocean_proximity'], dtype=np.float64)
test_set = pd.get_dummies(test_set, columns=['ocean_proximity'], dtype=np.float64)

# Ensure both sets have the same columns
missing_cols = set(train_set.columns) - set(test_set.columns)
for col in missing_cols:
    test_set[col] = 0
test_set = test_set[train_set.columns]

print("Encoded columns:", train_set.columns.tolist())

# ==========================
# 5. EDA (Exploratory Data Analysis)
# ==========================
def plot_histograms(df, bins=50, figsize=(16, 12), color='#1f77b4', title="Histogram of Numeric Features"):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        df[col].hist(ax=axes[i], bins=bins, color=color, edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, figsize=(16, 12), color='#2ca02c', title="Boxplots of Numeric Features"):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col], ax=axes[i], color=color)
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('Value')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, figsize=(12, 10), cmap="coolwarm", title="Correlation Heatmap"):
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, square=True, linewidths=0.5, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

print("\nPlotting Histograms and Boxplots for Training Set")
plot_histograms(train_set)
plot_boxplots(train_set)
plot_correlation_heatmap(train_set)

# Geospatial Plots
plt.figure(figsize=(10, 8))
plt.scatter(train_set.longitude, train_set.latitude, c=train_set['median_house_value'], cmap='jet', alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geospatial Distribution of House Values")
plt.colorbar(label="Median House Value")
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(train_set.longitude, train_set.latitude,
            c=train_set['median_house_value'],
            s=train_set['population']/100,
            cmap='jet', alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Geospatial Distribution of House Values (Bubble Size = Population/100)")
plt.colorbar(label="Median House Value")
plt.show()

# ==========================
# 6. Box-Cox Transformation (fit on train, apply on test)
# ==========================
boxcox_columns = [
    "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income", "median_house_value"
]

boxcox_lambdas = {}
for col in boxcox_columns:
    values = train_set[col].dropna()
    if (values <= 0).any():
        print(f"Skipped '{col}' (contains zero or negative values)")
        continue
    transformed_data, fitted_lambda = boxcox(values)
    train_set.loc[values.index, col] = transformed_data
    boxcox_lambdas[col] = fitted_lambda

def apply_boxcox_with_given_lambdas(df, lambdas, features):
    df = df.copy()
    for col in features:
        lmbda = lambdas.get(col)
        if lmbda is not None:
            values = df[col]
            if (values <= 0).any():
                print(f"Skipped '{col}' in test set (contains zero or negative values)")
                continue
            df[col] = boxcox_apply(values, lmbda)  # FIXED: pass lmbda as positional argument
    return df

test_set = apply_boxcox_with_given_lambdas(test_set, boxcox_lambdas, boxcox_columns)

print("\nBox-Cox Lambda Summary:")
for col in boxcox_lambdas:
    print(f"{col}: λ={boxcox_lambdas[col]:.4f}")

# ==========================
# 7. Outlier Removal
# ==========================
def remove_outliers_multiple_columns(df, columns=None, whis=3.5, min_outlier_count=1):
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_flags = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - whis * iqr
        upper = q3 + whis * iqr
        outlier_flags[col] = (df[col] < lower) | (df[col] > upper)
    outlier_counts = outlier_flags.sum(axis=1)
    to_drop = outlier_counts[outlier_counts >= min_outlier_count].index
    cleaned_df = df.drop(index=to_drop)
    print(f"Removed {len(to_drop)} rows with ≥{min_outlier_count} outliers.")
    return cleaned_df

selected_cols = [
    "housing_median_age", "total_rooms", "total_bedrooms", "population",
    "median_income", "median_house_value"
]
train_set_cleaned = remove_outliers_multiple_columns(train_set, columns=selected_cols, whis=3.5)
test_set_cleaned = remove_outliers_multiple_columns(test_set, columns=selected_cols, whis=3.5)

print("\nPlotting Cleaned Training Set Histogram")
plot_histograms(train_set_cleaned, title="Histogram after Outlier Removal (Train Set)")

# ==========================
# 8. Normalization
# ==========================
print("\n" + "="*25 + " NORMALIZATION " + "="*25)
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_set_cleaned), columns=train_set_cleaned.columns, index=train_set_cleaned.index)
test_scaled = pd.DataFrame(scaler.transform(test_set_cleaned), columns=test_set_cleaned.columns, index=test_set_cleaned.index)

print("Train normalized shape:", train_scaled.shape)
print("Test normalized shape:", test_scaled.shape)

print("\n" + "="*25 + " FINISHED " + "="*25)

## model selecation
# target is medain house value
train_scaled = np.array(train_scaled)
test_scaled = np.array(test_scaled)
y_train = train_scaled[:,8]
x_train = np.delete(train_scaled,8, axis=1)

y_test = test_scaled[:,8]
x_test = np.delete(test_scaled,8, axis=1)

model = SGDRegressor(random_state= 42)

lrs = np.logspace(-3, 0, 50)
alphas = np.logspace(-6, 0, 50)
param_distr = {'eta0':lrs, 'alpha':alphas}

random_serarch = RandomizedSearchCV(model, param_distributions=param_distr, n_iter=250, random_state=31)
random_serarch.fit(x_train, y_train.ravel())

print(random_serarch.best_params_, random_serarch.best_score_)

lr, alpha = random_serarch.best_params_.values()
d1,d2 = 0.1,0.1
lrs = np.r_[np.linspace((1-d1)*lr, (1+d1)*lr,50),lr]
alphas = np.r_[np.linspace((1-d2)*alpha, (1+d2)*alpha,50), alpha]

param_grid = {'eta0':lrs, 'alpha':alphas}

grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=5,refit= True)
grid_search.fit(x_train,y_train.ravel())

print(grid_search.best_params_, grid_search.best_score_)

model.score(x_test, y_test)