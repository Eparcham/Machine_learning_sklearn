import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ───────────────────────────────────────
# Load Data
# ───────────────────────────────────────
train_df = pd.read_csv('./data/energy-train-s.csv')
test_df = pd.read_csv('./data/energy-test-s.csv')

x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# ───────────────────────────────────────
# Create Pipeline
# ───────────────────────────────────────
degree = 4

pipeline = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree),
    SGDRegressor(
        max_iter=10000,
        tol=1e-4,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        learning_rate='constant',
        eta0=0.0001,
        random_state=42,
        verbose=1
    )
)

# ───────────────────────────────────────
# Fit Model
# ───────────────────────────────────────
pipeline.fit(x_train, y_train)

# ───────────────────────────────────────
# Predict & Evaluate
# ───────────────────────────────────────
y_pred = pipeline.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")
print(f"Test R²:  {r2:.4f}")

# ───────────────────────────────────────
# Residual Plot
# ───────────────────────────────────────
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
