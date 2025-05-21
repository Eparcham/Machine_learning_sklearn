import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/energy-train-s.csv')
test_df = pd.read_csv('./data/energy-test-s.csv')

x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

# ────────────────────────────────
# Polynomial Feature Expansion
# ────────────────────────────────
degree = 4
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
# Gradient Descent
# ────────────────────────────────
def gradient_descent(phi_train, y_train, phi_test, y_test, w, learning_rate=0.01, iterations=1000):
    n = len(y_train)
    train_mse_hist = []
    test_mse_hist = []
    train_r2_hist = []
    test_r2_hist = []

    for i in range(iterations):
        y_hat_train = phi_train @ w
        error = y_hat_train - y_train
        dw = (2 / n) * (phi_train.T @ error)
        w -= learning_rate * dw

        _, mse_train = compute_error(y_hat_train, y_train)
        train_r2 = r2(y_hat_train, y_train)
        y_hat_test = phi_test @ w
        _, mse_test = compute_error(y_hat_test, y_test)
        test_r2 = r2(y_hat_test, y_test)

        train_mse_hist.append(mse_train)
        test_mse_hist.append(mse_test)
        train_r2_hist.append(train_r2)
        test_r2_hist.append(test_r2)

        if i % 100 == 0 or i == iterations - 1:
            print(f"[Epoch {i+1:4d}] Train MSE: {mse_train:.5f}, Test MSE: {mse_test:.5f}, R2 Train: {train_r2:.4f}, R2 Test: {test_r2:.4f}")

    return w, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist

# ────────────────────────────────
# Train
# ────────────────────────────────
np.random.seed(0)
w_init = np.random.randn(phi_train.shape[1], 1) * 0.01
epochs = 5000
lr = 0.01

w_final, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist = gradient_descent(
    phi_train, y_train, phi_test, y_test, w_init, lr, epochs
)

# ────────────────────────────────
# Final Evaluation
# ────────────────────────────────
y_pred_test = phi_test @ w_final
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nFinal Test MAE: {mae_test:.4f}, Test MSE: {mse_test:.4f}, R2: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot Training and Test MSE
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
# Optional: Plot R2 over Epochs
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
