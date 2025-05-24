import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

# ───────────────────────────────────────────────
# Load Dataset
# ───────────────────────────────────────────────
TRAIN_CSV = "./data/energy-train-l.csv"
TEST_CSV = "./data/energy-test-l.csv"
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train_raw, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
X_test_raw, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

print(f"✅ Loaded data: Train = {len(y_train)}, Test = {len(y_test)}")

# ───────────────────────────────────────────────
# Polynomial Expansion
# ───────────────────────────────────────────────
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train = poly.fit_transform(X_train_raw)
X_test = poly.transform(X_test_raw)

feature_names = poly.get_feature_names_out([f'x{i}' for i in range(X_train_raw.shape[1])])
print(f"✅ Polynomial features: from {X_train_raw.shape[1]} to {X_train.shape[1]} features")

# ───────────────────────────────────────────────
# 1. GridSearchCV
# ───────────────────────────────────────────────
param_grid = {
    "eta0": np.logspace(-3, 0, 10),
    "alpha": np.logspace(-6, 0, 10),
}
grid_model = GridSearchCV(SGDRegressor(random_state=42), param_grid, cv=5, return_train_score=True)
grid_model.fit(X_train, y_train)

grid_results = pd.DataFrame(grid_model.cv_results_)
grid_results = grid_results.sort_values(by="mean_test_score", ascending=False)

print("🔍 Best (GridSearchCV):", grid_model.best_params_, f"R² = {grid_model.best_score_:.4f}")

# Plot Grid Search results
plt.figure(figsize=(6, 4))
plt.plot(grid_results['mean_test_score'].values[:10], label='Top 10 Test Scores')
plt.title("Top 10 GridSearchCV Mean Test Scores")
plt.ylabel("R²")
plt.xlabel("Rank")
plt.grid(True)
plt.legend()
plt.show()

# ───────────────────────────────────────────────
# 2. RandomizedSearchCV
# ───────────────────────────────────────────────
param_dist = {
    "eta0": np.logspace(-3, 0, 100),
    "alpha": np.logspace(-6, 0, 100),
}
random_model = RandomizedSearchCV(SGDRegressor(random_state=42), param_distributions=param_dist, n_iter=20, cv=5, random_state=42, verbose=0)
random_model.fit(X_train, y_train)

print("🔍 Best (RandomizedSearchCV):", random_model.best_params_, f"R² = {random_model.best_score_:.4f}")

# ───────────────────────────────────────────────
# 3. Hybrid Search (Random → Grid Refinement)
# ───────────────────────────────────────────────
# Use result from RandomizedSearch to define narrowed range for GridSearch
alpha_c = random_model.best_params_['alpha']
eta0_c = random_model.best_params_['eta0']

refined_alphas = np.logspace(np.log10(alpha_c/2), np.log10(alpha_c*2), 5)
refined_etas = np.logspace(np.log10(eta0_c/2), np.log10(eta0_c*2), 5)

refined_grid = {
    "alpha": refined_alphas,
    "eta0": refined_etas,
}

refined_model = GridSearchCV(SGDRegressor(random_state=42), refined_grid, cv=5, return_train_score=True)
refined_model.fit(X_train, y_train)

print("🔍 Best (Hybrid Search):", refined_model.best_params_, f"R² = {refined_model.best_score_:.4f}")

# ───────────────────────────────────────────────
# Final Report
# ───────────────────────────────────────────────
print("\n📊 Summary of Search Strategies")
print("-" * 45)
print(f"{'Method':<20} {'Best R²':<10} {'Best Params'}")
print("-" * 45)
print(f"{'GridSearchCV':<20} {grid_model.best_score_:.4f} {grid_model.best_params_}")
print(f"{'RandomSearchCV':<20} {random_model.best_score_:.4f} {random_model.best_params_}")
print(f"{'HybridSearch':<20} {refined_model.best_score_:.4f} {refined_model.best_params_}")

# Optional: Evaluate best model from Hybrid on test set
final_model = refined_model.best_estimator_
y_pred = final_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
print(f"\n✅ Final Test R²: {test_r2:.4f}")
