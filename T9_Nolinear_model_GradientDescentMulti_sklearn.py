import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor

# ────────────────────────────────
# Load Data
# ────────────────────────────────
train_df = pd.read_csv('./data/energy-train-s.csv')
test_df = pd.read_csv('./data/energy-test-s.csv')

x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.ravel()
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.ravel()

# ────────────────────────────────
# Scaling
# ────────────────────────────────
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ────────────────────────────────
# Polynomial Features
# ────────────────────────────────
degree = 3
poly = PolynomialFeatures(degree)
phi_train = poly.fit_transform(x_train)
phi_test = poly.transform(x_test)

# ────────────────────────────────
# Error Metrics
# ────────────────────────────────
def compute_error(y_hat, y_true):
    error = y_hat - y_true
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    return mae, mse

def r2(y_hat, y_true):
    ss_res = np.sum((y_hat - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ────────────────────────────────
# Training Function
# ────────────────────────────────
def train_mode(phi_train, y_train, phi_test, y_test, iterations=1000):
    model = SGDRegressor(eta0=0.0001)

    train_mse_hist, test_mse_hist = [], []
    train_r2_hist, test_r2_hist = [], []

    for i in range(iterations):
        model.partial_fit(phi_train, y_train)

        y_hat_train = model.predict(phi_train)
        y_hat_test = model.predict(phi_test)

        _, mse_train = compute_error(y_hat_train, y_train)
        _, mse_test = compute_error(y_hat_test, y_test)
        r2_train = r2(y_hat_train, y_train)
        r2_test = r2(y_hat_test, y_test)

        train_mse_hist.append(mse_train)
        test_mse_hist.append(mse_test)
        train_r2_hist.append(r2_train)
        test_r2_hist.append(r2_test)

        if i % 100 == 0 or i == iterations - 1:
            print(f"[Epoch {i+1:4d}] Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}, R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")

    return model, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist

# ────────────────────────────────
# Train Model
# ────────────────────────────────
epochs = 20000
model, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist = train_mode(
    phi_train, y_train, phi_test, y_test, iterations=epochs
)

# ────────────────────────────────
# Final Evaluation
# ────────────────────────────────
y_pred_test = model.predict(phi_test)
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nFinal Test MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, R2: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot MSE
# ────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(train_mse_hist, label="Train MSE", linewidth=2)
plt.plot(test_mse_hist, label="Test MSE", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Test MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ────────────────────────────────
# Plot R²
# ────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(train_r2_hist, label="Train R²", linewidth=2)
plt.plot(test_r2_hist, label="Test R²", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Training vs Test R²")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ────────────────────────────────
# Optional: Plot Fit Line on Test Set (only if 1 feature)
# ────────────────────────────────
if x_test.shape[1] == 1:
    x_axis = x_test[:, 0]
    x_sorted_idx = np.argsort(x_axis)
    y_true_sorted = y_test[x_sorted_idx]
    y_pred_sorted = y_pred_test[x_sorted_idx]
    x_sorted = x_axis[x_sorted_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_sorted, y_true_sorted, label='Actual', color='blue', alpha=0.6)
    plt.plot(x_sorted, y_pred_sorted, label='Predicted', color='red', linewidth=2)
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Target")
    plt.title("Model Fit on Test Data (1D)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ────────────────────────────────
# Residual Plot
# ────────────────────────────────
residuals = y_test - y_pred_test
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_test, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot on Test Data")
plt.grid(True)
plt.tight_layout()
plt.show()
