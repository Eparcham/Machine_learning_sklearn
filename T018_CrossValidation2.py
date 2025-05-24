import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# ───────────────────────────────────────────────
# Load Data
# ───────────────────────────────────────────────
TRAIN_CSV = "./data/energy-train-l.csv"
TEST_CSV = "./data/energy-test-l.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train_raw, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
X_test_raw, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# ───────────────────────────────────────────────
# Polynomial Feature Expansion (Degree = 3)
# ───────────────────────────────────────────────
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train = poly.fit_transform(X_train_raw)
X_test = poly.transform(X_test_raw)

# ───────────────────────────────────────────────
# HOLD-OUT SPLIT VISUALIZATION
# ───────────────────────────────────────────────
x_train, x_valid, y_train_split, y_valid_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=2
)

plt.figure(figsize=(10, 4))
plt.scatter(x_train[:, 0], y_train_split, color="red", label="Train (Hold-Out)")
plt.scatter(x_valid[:, 0], y_valid_split, color="blue", label="Validation (Hold-Out)")
plt.scatter(X_test[:, 0], y_test, color="green", label="Test Set")
plt.legend()
plt.title("Hold-Out Split Visualization")
plt.grid(True)
plt.show()

# ───────────────────────────────────────────────
# K-FOLD CROSS-VALIDATION (Linear Regression)
# ───────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=2)
kf_scores = []

for i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
    X_kf_train, X_kf_valid = X_train[train_idx], X_train[valid_idx]
    y_kf_train, y_kf_valid = y_train[train_idx], y_train[valid_idx]

    model = LinearRegression()
    model.fit(X_kf_train, y_kf_train)

    score = model.score(X_kf_valid, y_kf_valid)
    kf_scores.append(score)
    print(f"K-Fold {i+1} R²: {score:.4f}")

print(f"Average K-Fold R²: {np.mean(kf_scores):.4f}")

# ───────────────────────────────────────────────
# Final Evaluation on Hold-Out & Test for Each Model
# ───────────────────────────────────────────────
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10_000),
    "Ridge Regression": Ridge(alpha=1.0)
}

for name, model in models.items():
    model.fit(x_train, y_train_split)
    r2_val = model.score(x_valid, y_valid_split)
    r2_test = model.score(X_test, y_test)
    y_test_pred = model.predict(X_test)

    print(f"\n{name}")
    print(f"Validation R²: {r2_val:.4f}")
    print(f"Test R²:       {r2_test:.4f}")

    # Optional: plot predictions
    plt.figure(figsize=(6, 3))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{name} - Test Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
