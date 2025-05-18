import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/auto-train-preprocessed.csv')
test_df = pd.read_csv('./data/auto-test-preprocessed.csv')

x_train = train_df.iloc[:, :-1].values.flatten()
y_train = train_df.iloc[:, -1].values.flatten()
x_test = test_df.iloc[:, :-1].values.flatten()
y_test = test_df.iloc[:, -1].values.flatten()

# ────────────────────────────────
# Polynomial Model and Error Calculation
# ────────────────────────────────
def PolynomialRegression(x, w, degree):
    phi = np.vstack([x**i for i in range(degree + 1)]).T  # shape: (n_samples, degree+1)
    return phi @ w, phi

def compute_error(y_hat, y_true):
    error = y_hat.flatten() - y_true.flatten()
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    return mae, mse

def r2(y_hat, y_true):
    ss_res = np.sum((y_hat.flatten() - y_true.flatten()) ** 2)
    ss_tot = np.sum((y_true.flatten() - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ────────────────────────────────
# Gradient Descent Implementation
# ────────────────────────────────
def gradient_descent(x, y, w, learning_rate=0.01, iterations=1000, degree=3):
    n = len(x)
    mse_history = []

    for i in range(iterations):
        y_hat, phi = PolynomialRegression(x, w, degree)
        error = y_hat - y.reshape(-1, 1)  # shape: (n,1)
        dw = (2 / n) * (phi.T @ error)  # shape: (degree+1, 1)
        w -= learning_rate * dw

        # Record MSE
        _, mse = compute_error(y_hat, y)
        mse_history.append(mse)

        if i % 100 == 0 or i == iterations - 1:
            print(f"[Epoch {i+1:4d}] MSE: {mse:.5f} R2: {r2(y_hat, y):.4f}")

    return w, mse_history

# ────────────────────────────────
# Initialize and Run Training
# ────────────────────────────────
np.random.seed(0)
degree = 2
w_init = np.random.randn(degree + 1, 1) * 0.01
epochs = 10000
lr = 0.1

w_final, mse_hist = gradient_descent(x_train, y_train, w_init, lr, epochs, degree)

# ────────────────────────────────
# Evaluation on Test Set
# ────────────────────────────────
y_pred_test, _ = PolynomialRegression(x_test, w_final, degree)
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nTest MAE: {mae_test:.4f}, Test MSE: {mse_test:.4f}, R2: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot Results
# ────────────────────────────────
plt.figure(figsize=(10, 4))

# Plot regression curve
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, label="Train Data", alpha=0.7)
x_sorted = np.linspace(min(x_train), max(x_train), 300)
y_curve, _ = PolynomialRegression(x_sorted, w_final, degree)
plt.plot(x_sorted, y_curve, color='red', label='Fitted Curve')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title(f"Polynomial Regression (Degree {degree})")
plt.legend()
plt.grid(True)

# Plot MSE over epochs
plt.subplot(1, 2, 2)
plt.plot(mse_hist, label="Training MSE", color='blue')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE Over Training")
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.show()
