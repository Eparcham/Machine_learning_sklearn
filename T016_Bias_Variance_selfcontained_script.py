import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import r2_score

# ───────────────────────────────────────────────
# Load Data
# ───────────────────────────────────────────────
TRAIN_CSV = "./data/energy-train-l.csv"
TEST_CSV = "./data/energy-test-l.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

# ───────────────────────────────────────────────
# HOLD-OUT SPLIT
# ───────────────────────────────────────────────
x_train, x_valid, y_train_split, y_valid_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=2
)

plt.figure(figsize=(10, 4))
plt.scatter(x_train, y_train_split, color="red", label="Train (Hold-Out)")
plt.scatter(x_valid, y_valid_split, color="blue", label="Validation (Hold-Out)")
plt.scatter(X_test, y_test, color="green", label="Test Set")
plt.legend()
plt.title("Hold-Out Split Visualization")
plt.grid(True)
plt.show()

# ───────────────────────────────────────────────
# K-FOLD CROSS-VALIDATION
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
    print(f"K-Fold {i + 1} R²: {score:.4f}")

print(f"Average K-Fold R²: {np.mean(kf_scores):.4f}")

# ───────────────────────────────────────────────
# LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# ───────────────────────────────────────────────

loo = LeaveOneOut()
y_preds = []

for train_idx, valid_idx in loo.split(X_train):
    X_loo_train, X_loo_valid = X_train[train_idx], X_train[valid_idx]
    y_loo_train, y_loo_valid = y_train[train_idx], y_train[valid_idx]

    model = LinearRegression()
    model.fit(X_loo_train, y_loo_train)
    y_pred = model.predict(X_loo_valid)
    y_preds.append(y_pred[0])  # Save scalar prediction

# Compute R² across all predictions and ground truth
loo_r2 = r2_score(y_train, y_preds)
print(f"Average LOOCV R²: {loo_r2:.4f}")


# ───────────────────────────────────────────────
# FINAL MODEL ON HOLD-OUT SPLIT
# ───────────────────────────────────────────────
final_model = LinearRegression()
final_model.fit(x_train, y_train_split)

val_score = final_model.score(x_valid, y_valid_split)
test_score = final_model.score(X_test, y_test)

print(f"Hold-Out Validation R²: {val_score:.4f}")
print(f"Test Set R²: {test_score:.4f}")
