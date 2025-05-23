import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
TRAIN_CSV   = "./data/energy-train-s.csv"
TEST_CSV    = "./data/energy-test-s.csv"
DEGREE_RANGE = range(1, 11)        # polynomial degrees 1 … 10
NUM_ROUNDS   = 100                 # bootstrap repetitions
SEED         = 42
rng          = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

# containers for results
biases, variances, mses = [], [], []

# ----------------------------------------------------------------------
# bias–variance decomposition by simulation
# ----------------------------------------------------------------------
for d in DEGREE_RANGE:
    preds = np.zeros((NUM_ROUNDS, len(y_test)))

    for r in range(NUM_ROUNDS):
        # bootstrap sample from the training data
        X_boot, y_boot = resample(
            X_train, y_train,
            replace=True,
            n_samples=len(X_train),
            random_state=SEED + r,
        )

        # pipeline for this degree
        poly   = PolynomialFeatures(degree=d, include_bias=True)
        model  = LinearRegression()

        X_boot_poly = poly.fit_transform(X_boot)
        X_test_poly = poly.transform(X_test)

        model.fit(X_boot_poly, y_boot)
        preds[r] = model.predict(X_test_poly)

    # statistics across rounds
    pred_mean  = preds.mean(axis=0)
    bias_sq    = np.mean((pred_mean - y_test) ** 2)
    variance   = np.mean(preds.var(axis=0, ddof=0))
    mse        = bias_sq + variance           # noise term omitted

    biases.append(bias_sq)
    variances.append(variance)
    mses.append(mse)
    print(f"degree is {d} bias is {bias_sq}, variance is {variance} mse is {mse}")

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
