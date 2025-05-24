import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression , SGDRegressor

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/auto-train-multi.csv')
test_df = pd.read_csv('./data/auto-test-multi.csv')

x_train = train_df.iloc[:, :-1].values  # shape: (n_samples, n_features)
y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

# ────────────────────────────────
# Polynomial Feature Expansion
# ────────────────────────────────
degree = 3

model_pipline = make_pipeline(PolynomialFeatures(degree),
              LinearRegression(),
              )

model_pipline.fit(x_train, y_train)
print(model_pipline.score(x_train, y_train))
print(model_pipline.score(x_test, y_test))
