import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
TRAIN_CSV = "./data/energy-train-s.csv"
TEST_CSV  = "./data/energy-test-s.csv"
DEGREE_RANGE = range(1, 11)      # 1 → 10
NUM_ROUNDS   = 100               # bootstrap rounds for bias-variance
SEED         = 42

# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

# ----------------------------------------------------------------------
# bias–variance decomposition across degrees
# ----------------------------------------------------------------------
biases, variances, mses = [], [], []

for d in DEGREE_RANGE:
    poly = PolynomialFeatures(degree=d, include_bias=True)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly  = poly.transform(X_test)

    model = LinearRegression()

    mse, bias, var = bias_variance_decomp(
        model,
        X_train_poly, y_train,
        X_test_poly,  y_test,
        loss="mse",
        num_rounds=NUM_ROUNDS,
        random_seed=SEED,
    )

    biases.append(bias)       # bias already returned as bias²
    variances.append(var)
    mses.append(mse)
    print(f"degree is {d} bias is {bias}, variance is {var} mse is {var+bias}")

# ----------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(DEGREE_RANGE, biases,    marker="o", label="Bias²")
plt.plot(DEGREE_RANGE, variances, marker="s", label="Variance")
plt.plot(DEGREE_RANGE, mses,      marker="^", label="MSE (Bias² + Var)")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title("Bias–Variance trade-off on Energy dataset")
plt.legend()
plt.tight_layout()
plt.show()
