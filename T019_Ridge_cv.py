import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
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

print(f"✅ Loaded data: Train size = {len(y_train)}, Test size = {len(y_test)}")

# ───────────────────────────────────────────────
# Polynomial Feature Expansion (Degree = 3)
# ───────────────────────────────────────────────
degree = 5
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train = poly.fit_transform(X_train_raw)
X_test = poly.transform(X_test_raw)

# Get the names of generated features
feature_names = poly.get_feature_names_out(input_features=[f'x{i}' for i in range(X_train_raw.shape[1])])

# Display summary
print(f"✅ Polynomial features applied (degree = {degree})")
print(f"   Input features expanded from {X_train_raw.shape[1]} to {X_train.shape[1]}")
print("\n🔍 Generated polynomial terms:")
for i, name in enumerate(feature_names):
    print(f"{i+1:3d}: {name}")

# ───────────────────────────────────────────────
# RidgeCV: Cross-Validated Ridge Regression
# ───────────────────────────────────────────────
alphas = (1, 0.1, 0.01, 1.5, 2, 2.5,3, 3.5, 0.001)
ridge_cv = RidgeCV(alphas=alphas, scoring="r2", cv=10)
ridge_cv.fit(X_train, y_train)

best_alpha = ridge_cv.alpha_
print(f"\n🔍 Best alpha from RidgeCV: {best_alpha}")

# Re-train Ridge with best alpha
model = Ridge(alpha=best_alpha)
model.fit(X_train, y_train)

# ───────────────────────────────────────────────
# Evaluation on Test Set
# ───────────────────────────────────────────────
y_test_pred = model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)

print(f"📈 Test R² score: {r2_test:.4f}")

# ───────────────────────────────────────────────
# Optional: Visualization
# ───────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_test_pred, alpha=0.7, color="dodgerblue", edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression (Test Set Predictions)")
plt.grid(True)
plt.tight_layout()
plt.show()
