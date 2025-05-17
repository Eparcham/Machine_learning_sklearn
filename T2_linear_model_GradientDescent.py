import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/student_loan_train.csv')
test_df = pd.read_csv('./data/student_loan_test.csv')

x_train = train_df.iloc[:, 0].values
y_train = train_df.iloc[:, 1].values
x_test = test_df.iloc[:, 0].values
y_test = test_df.iloc[:, 1].values

# ────────────────────────────────
# Linear Model and Error Calculation
# ────────────────────────────────
def linear_model(x, w0, w1):
    return w1 * x + w0

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
# Gradient Descent Implementation
# ────────────────────────────────
def gradient_descent(x, y, w0, w1, learning_rate=0.01, iterations=1000):
    n = len(x)
    mse_history = []

    for i in range(iterations):
        y_hat = linear_model(x, w0, w1)
        error = y_hat - y

        # Gradients
        dw1 = (2/n) * np.dot(error, x)
        dw0 = (2/n) * np.sum(error)

        # Update weights
        w1 -= learning_rate * dw1
        w0 -= learning_rate * dw0

        # Record MSE
        _, mse = compute_error(y_hat, y)
        mse_history.append(mse)
        r2_error = r2(y_hat, y)
        if i % 100 == 0 or i == iterations - 1:
            print(f"[Epoch {i+1:4d}] MSE: {mse:.5f}, w1: {w1:.5f}, w0: {w0:.5f} r2_error: {r2_error:.4f}")

    return w0, w1, mse_history

# ────────────────────────────────
# Initialize and Run Training
# ────────────────────────────────
np.random.seed(0)
w0_init = np.random.rand()
w1_init = np.random.rand()
epochs = 1000
lr = 0.01

w0_final, w1_final, mse_hist = gradient_descent(x_train, y_train, w0_init, w1_init, lr, epochs)

# ────────────────────────────────
# Evaluation on Test Set
# ────────────────────────────────
y_pred_test = linear_model(x_test, w0_final, w1_final)
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nTest MAE: {mae_test:.4f}, Test MSE: {mse_test:.4f} R2 error: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot Results
# ────────────────────────────────
plt.figure(figsize=(10, 4))

# Plot regression line
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, label="Train Data", alpha=0.7)
plt.plot(x_train, linear_model(x_train, w0_final, w1_final), color='red', label='Fitted Line')
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Fit")
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
