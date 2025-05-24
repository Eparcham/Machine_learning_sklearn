import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/auto-train-multi.csv')
test_df = pd.read_csv('./data/auto-test-multi.csv')

x_train = train_df.iloc[:, :-1].values  # shape: (n_samples, n_features)
y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

# ────────────────────────────────
# Polynomial Feature Expansion
# ────────────────────────────────
degree = 3
poly = PolynomialFeatures(degree)
phi_train = poly.fit_transform(x_train)  # shape: (n_samples, n_poly_features)
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
def gradient_descent(phi, y, w, learning_rate=0.01, iterations=1000):
    n = len(y)
    mse_history = []

    for i in range(iterations):
        y_hat = phi @ w
        error = y_hat - y
        dw = (2 / n) * (phi.T @ error)
        w -= learning_rate * dw

        _, mse = compute_error(y_hat, y)
        mse_history.append(mse)

        if i % 100 == 0 or i == iterations - 1:
            print(f"[Epoch {i+1:4d}] MSE: {mse:.5f}, R2: {r2(y_hat, y):.4f}")

    return w, mse_history

# ────────────────────────────────
# Train
# ────────────────────────────────
np.random.seed(0)
w_init = np.random.randn(phi_train.shape[1], 1) * 0.01
epochs = 5000
lr = 0.1

w_final, mse_hist = gradient_descent(phi_train, y_train, w_init, lr, epochs)

# ────────────────────────────────
# Evaluate
# ────────────────────────────────
y_pred_test = phi_test @ w_final
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nTest MAE: {mae_test:.4f}, Test MSE: {mse_test:.4f}, R2: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot MSE Curve
# ────────────────────────────────
plt.figure(figsize=(6, 4))
plt.plot(mse_hist, color='blue', label="Training MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE Over Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
