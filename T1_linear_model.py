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


# Define the function f(x) = x^2
def func_x2(x):
    return x**2

# Define the gradient of f(x)
def gradfuncx2(x):
    return 2 * x

# Setup for plotting
x_vals = np.linspace(-5, 5, 400)
y_vals = func_x2(x_vals)

# Gradient descent parameters
xi = -4           # initial value
alpha = 0.1       # learning rate
iterations = 20   # number of steps

# To store values
path_x = [xi]
path_y = [func_x2(xi)]

# Perform gradient descent
for i in range(iterations):
    grad = gradfuncx2(xi)
    xi = xi - alpha * grad
    path_x.append(xi)
    path_y.append(func_x2(xi))
    print(f"Step {i+1:02d}: x = {xi:.6f}, f(x) = {path_y[-1]:.6f}, grad = {grad:.6f}")

# Plot function and gradient descent steps
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label=r"$f(x) = x^2$", linewidth=2)
plt.scatter(path_x, path_y, color="red", label="Descent steps", zorder=5)
plt.plot(path_x, path_y, color="red", linestyle="--", alpha=0.6)

plt.title("Gradient Descent on $f(x) = x^2$")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Define the 3D function
def func_3d(x, y):
    return x**2 + y**2

# Define the gradient of the function
def gradfunc3d(x, y):
    return 2 * x, 2 * y

# Create a meshgrid for plotting the surface
x_vals, y_vals = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
z_vals = func_3d(x_vals, y_vals)

# Initialize descent parameters
xi, yi = 4.0, 4.0          # Initial point
alpha = 0.9                # Learning rate
iterations = 30            # Number of steps

# Track descent path
path_x = [xi]
path_y = [yi]
path_z = [func_3d(xi, yi)]

# Gradient Descent Loop
for i in range(iterations):
    grad_x, grad_y = gradfunc3d(xi, yi)
    xi -= alpha * grad_x
    yi -= alpha * grad_y
    zi = func_3d(xi, yi)

    path_x.append(xi)
    path_y.append(yi)
    path_z.append(zi)

    print(f"Step {i+1:02d}: x = {xi:.6f}, y = {yi:.6f}, f(x, y) = {zi:.6f}, grad = ({grad_x:.6f}, {grad_y:.6f})")

# ──────────────────── 3D Plot ──────────────────────
fig = plt.figure(figsize=(10, 5))

# 3D surface + descent path
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.plot_surface(x_vals, y_vals, z_vals, cmap='viridis', alpha=0.8)
ax3d.plot(path_x, path_y, path_z, color='red', marker='o', label="Descent Path")
ax3d.set_title("3D Surface and Gradient Descent")
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("f(x, y)")
ax3d.legend()

# ──────────────────── Contour Plot ──────────────────────
ax2d = fig.add_subplot(1, 2, 2)
contours = ax2d.contour(x_vals, y_vals, z_vals, levels=50, cmap='viridis')
ax2d.plot(path_x, path_y, color='red', marker='o', label="Descent Path")
ax2d.set_title("Contour Plot of $f(x, y) = x^2 + y^2$")
ax2d.set_xlabel("x")
ax2d.set_ylabel("y")
ax2d.legend()
ax2d.grid(True)

plt.tight_layout()
plt.show()


