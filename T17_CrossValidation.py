import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
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
# HOLD-OUT SPLIT VISUALIZATION
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
# K-FOLD SPLIT VISUALIZATION (For One Fold)
# ───────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=2)
fold = 4
train_idx, valid_idx = list(kf.split(X_train))[fold]

plt.figure(figsize=(8, 4))
plt.scatter(X_train[train_idx], y_train[train_idx], color="red", label="Train (Fold)")
plt.scatter(X_train[valid_idx], y_train[valid_idx], color="blue", label="Validation (Fold)")
plt.legend()
plt.title(f"K-Fold Visualization - Fold {fold + 1}")
plt.grid(True)
plt.show()

# ───────────────────────────────────────────────
# K-FOLD CROSS-VALIDATION TRAINING
# ───────────────────────────────────────────────
kf_scores = []
for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_valid = X_train[train_index], X_train[valid_index]
    y_fold_train, y_fold_valid = y_train[train_index], y_train[valid_index]

    model = LinearRegression()
    model.fit(X_fold_train, y_fold_train)

    score = model.score(X_fold_valid, y_fold_valid)
    kf_scores.append(score)
    print(f"Fold {i + 1} R² score: {score:.4f}")

print(f"Average K-Fold R² score: {np.mean(kf_scores):.4f}")

# ───────────────────────────────────────────────
# FINAL MODEL ON HOLD-OUT
# ───────────────────────────────────────────────
final_model = LinearRegression()
final_model.fit(x_train, y_train_split)

valid_score = final_model.score(x_valid, y_valid_split)
test_score = final_model.score(X_test, y_test)

print(f"Hold-Out Validation R²: {valid_score:.4f}")
print(f"Test Set R²: {test_score:.4f}")
