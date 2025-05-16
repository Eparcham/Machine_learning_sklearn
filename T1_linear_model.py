import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/student_loan_train.csv')
test_df = pd.read_csv('./data/student_loan_test.csv')

print("Train set preview:\n", train_df.head())
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

x_train = train_df.iloc[:, 0].values
y_train = train_df.iloc[:, 1].values

x_test = test_df.iloc[:, 0].values
y_test = test_df.iloc[:, 1].values

# ────────────────────────────────
# Optional: Visualize Training Data
# ────────────────────────────────
VISUALIZE = False
if VISUALIZE:
    plt.figure()
    plt.scatter(x_train, y_train, label="Train Data")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Training Data Scatter Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

# ────────────────────────────────
# Linear Model Function
# ────────────────────────────────
def linear_model(x, w0, w1, y):
    """
    Predicts using linear model and computes MAE and MSE.

    Args:
        x (array-like): Input features
        w0 (float): Intercept
        w1 (float): Slope
        y (array-like): Ground-truth values

    Returns:
        y_hat (ndarray): Predictions
        mae (float): Mean Absolute Error
        mse (float): Mean Squared Error
    """
    x = np.asarray(x)
    y = np.asarray(y)

    y_hat = w1 * x + w0
    error = y_hat - y
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    return y_hat, mae, mse

# ────────────────────────────────
# Initial Random Parameters
# ────────────────────────────────
np.random.seed(0)
w0_init = np.random.rand()
w1_init = np.random.rand()

y_hat, mae, mse = linear_model(x_train, w0_init, w1_init, y_train)
print(f"Initial Random MAE: {mae:.4f}, MSE: {mse:.4f}")

plt.figure()
plt.scatter(x_train, y_train, label="Train Data")
plt.plot(x_train, y_hat, 'r', label="Prediction (Random Weights)")
plt.title("Initial Model Fit")
plt.legend()
plt.grid(True)
plt.show()

# ────────────────────────────────
# Search Optimal w1 (w0 = 0)
# ────────────────────────────────
w0 = 0
w1_range = np.linspace(-100, 250, 5000)

mae_list = []
mse_list = []

for w1 in w1_range:
    _, mae, mse = linear_model(x_train, w0, w1, y_train)
    mae_list.append(mae)
    mse_list.append(mse)

mae_list = np.array(mae_list)
mse_list = np.array(mse_list)

best_mae_index = np.argmin(mae_list)
best_mse_index = np.argmin(mse_list)

print(f"[Best MAE]  w1 = {w1_range[best_mae_index]:.4f} → MAE = {mae_list[best_mae_index]:.4f}")
print(f"[Best MSE]  w1 = {w1_range[best_mse_index]:.4f} → MSE = {mse_list[best_mse_index]:.4f}")

# ────────────────────────────────
# Plot Loss Curves
# ────────────────────────────────
plt.figure()
plt.plot(w1_range, mae_list, color="red")
plt.title("MAE vs. w1")
plt.xlabel("w1")
plt.ylabel("MAE")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(w1_range, mse_list, color="blue")
plt.title("MSE vs. w1")
plt.xlabel("w1")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────
# Load Training and Testing Data
# ────────────────────────────────
train_df = pd.read_csv('./data/student_loan_train.csv')
test_df = pd.read_csv('./data/student_loan_test.csv')

print("Train set preview:\n", train_df.head())
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

x_train = train_df.iloc[:, 0].values
y_train = train_df.iloc[:, 1].values

x_test = test_df.iloc[:, 0].values
y_test = test_df.iloc[:, 1].values

# ────────────────────────────────
# Optional: Visualize Training Data
# ────────────────────────────────
VISUALIZE = False
if VISUALIZE:
    plt.figure()
    plt.scatter(x_train, y_train, label="Train Data")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Training Data Scatter Plot")
    plt.legend()
    plt.grid(True)
    plt.show()

# ────────────────────────────────
# Linear Model Function
# ────────────────────────────────
def linear_model(x, w0, w1, y):
    """
    Predicts using linear model and computes MAE and MSE.

    Args:
        x (array-like): Input features
        w0 (float): Intercept
        w1 (float): Slope
        y (array-like): Ground-truth values

    Returns:
        y_hat (ndarray): Predictions
        mae (float): Mean Absolute Error
        mse (float): Mean Squared Error
    """
    x = np.asarray(x)
    y = np.asarray(y)

    y_hat = w1 * x + w0
    error = y_hat - y
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    return y_hat, mae, mse

# ────────────────────────────────
# Initial Random Parameters
# ────────────────────────────────
np.random.seed(0)
w0_init = np.random.rand()
w1_init = np.random.rand()

y_hat, mae, mse = linear_model(x_train, w0_init, w1_init, y_train)
print(f"Initial Random MAE: {mae:.4f}, MSE: {mse:.4f}")

plt.figure()
plt.scatter(x_train, y_train, label="Train Data")
plt.plot(x_train, y_hat, 'r', label="Prediction (Random Weights)")
plt.title("Initial Model Fit")
plt.legend()
plt.grid(True)
plt.show()

# ────────────────────────────────
# Search Optimal w1 (w0 = 0)
# ────────────────────────────────
w0 = 0
w1_range = np.linspace(-100, 250, 5000)

mae_list = []
mse_list = []

for w1 in w1_range:
    _, mae, mse = linear_model(x_train, w0, w1, y_train)
    mae_list.append(mae)
    mse_list.append(mse)

mae_list = np.array(mae_list)
mse_list = np.array(mse_list)

best_mae_index = np.argmin(mae_list)
best_mse_index = np.argmin(mse_list)

print(f"[Best MAE]  w1 = {w1_range[best_mae_index]:.4f} → MAE = {mae_list[best_mae_index]:.4f}")
print(f"[Best MSE]  w1 = {w1_range[best_mse_index]:.4f} → MSE = {mse_list[best_mse_index]:.4f}")

# ────────────────────────────────
# Plot Loss Curves
# ────────────────────────────────
plt.figure()
plt.plot(w1_range, mae_list, color="red")
plt.title("MAE vs. w1")
plt.xlabel("w1")
plt.ylabel("MAE")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(w1_range, mse_list, color="blue")
plt.title("MSE vs. w1")
plt.xlabel("w1")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

