import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# ────────────────────────────────
# Load and Prepare Data
# ────────────────────────────────
df = pd.read_csv('./data/exam.csv')  # Make sure your CSV file is correct

train, test = train_test_split(df, train_size=0.8, test_size=0.2, random_state=3)
x_train = train.iloc[:, :-1].values  # (n_samples, 1)
y_train = train.iloc[:, -1].values.reshape(-1, 1)

x_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values.reshape(-1, 1)

# Normalize the feature
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Add bias term (column of 1s)
x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])

# ────────────────────────────────
# Logistic Regression Functions
# ────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(x, w):
    return sigmoid(x @ w)

def bce(y, y_hat):
    epsilon = 1e-10
    return -np.mean(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))

def gradient(x, y, y_hat):
    return (x.T @ (y_hat - y)) / len(y)

def accuracy(y, y_hat):
    return np.mean(y == np.round(y_hat))

# ────────────────────────────────
# Gradient Descent with Best Weight Tracking
# ────────────────────────────────
def gradient_descent(x_train, y_train, x_test, y_test, w_init, learning_rate=0.01, iterations=1000):
    w = w_init.copy()
    train_losses = []
    test_losses = []
    best_w = None
    best_test_loss = float('inf')

    for i in range(iterations):
        y_hat_train = logistic_regression(x_train, w)
        train_loss = bce(y_train, y_hat_train)
        w -= learning_rate * gradient(x_train, y_train, y_hat_train)

        y_hat_test = logistic_regression(x_test, w)
        test_loss = bce(y_test, y_hat_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_w = w.copy()

        if i % 100 == 0 or i == iterations - 1:
            acc_train = accuracy(y_train, y_hat_train)
            acc_test = accuracy(y_test, y_hat_test)
            print(f"[Epoch {i+1:4d}] Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f} | Train Acc: {acc_train:.4f} | Test Acc: {acc_test:.4f}")

    return best_w, train_losses, test_losses

# ────────────────────────────────
# Run Training
# ────────────────────────────────
np.random.seed(0)
w_init = np.random.randn(2, 1) * 0.01  # [bias, weight]
epochs = 10000
lr = 0.1

w_best, loss_train, loss_test = gradient_descent(x_train, y_train, x_test, y_test, w_init, lr, epochs)

# ────────────────────────────────
# Final Evaluation
# ────────────────────────────────
y_pred_test = logistic_regression(x_test, w_best)
test_acc = accuracy(y_test, y_pred_test)

print("\nBest Weights:")
print(f"w0 (bias):    {w_best[0][0]:.5f}")
print(f"w1 (feature): {w_best[1][0]:.5f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

# ────────────────────────────────
# Plot Losses
# ────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Training vs Test Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()