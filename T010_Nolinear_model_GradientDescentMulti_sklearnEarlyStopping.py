import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
import joblib

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
# Training Function with Early Stopping
# ────────────────────────────────
def train_model(phi_train, y_train, phi_test, y_test, iterations=1000, patience=5):
    model = SGDRegressor(eta0=0.0001, learning_rate='constant', random_state=2)

    train_mse_hist, test_mse_hist = [], []
    train_r2_hist, test_r2_hist = [], []

    best_loss = np.inf
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(iterations):
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

        if mse_test < best_loss:
            best_loss = mse_test
            best_epoch = epoch
            early_stop_counter = 0
            joblib.dump(model, './data/best_model.pkl')
        else:
            early_stop_counter += 1
            if early_stop_counter > patience:
                print(f"\n[Early Stop] Stopping at epoch {epoch} with best loss {best_loss:.4f}")
                break

        if epoch % 100 == 0 or epoch == iterations - 1:
            print(f"[Epoch {epoch+1:4d}] Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}, R2 Train: {r2_train:.4f}, R2 Test: {r2_test:.4f}")

    return best_epoch, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist

# ────────────────────────────────
# Train Model
# ────────────────────────────────
epochs = 10000
best_epoch, train_mse_hist, test_mse_hist, train_r2_hist, test_r2_hist = train_model(
    phi_train, y_train, phi_test, y_test, iterations=epochs, patience=5
)

# ────────────────────────────────
# Load and Evaluate Best Model
# ────────────────────────────────
best_model = joblib.load('./data/best_model.pkl')
y_pred_test = best_model.predict(phi_test)
mae_test, mse_test = compute_error(y_pred_test, y_test)
print(f"\nFinal Test MAE: {mae_test:.4f}, MSE: {mse_test:.4f}, R2: {r2(y_pred_test, y_test):.4f}")

# ────────────────────────────────
# Plot MSE with Early Stopping
# ────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(train_mse_hist, label="Train MSE")
plt.plot(test_mse_hist, label="Test MSE")
plt.axvline(best_epoch, color='red', linestyle='--', label=f"Early Stop @ {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training vs Test MSE with Early Stopping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ────────────────────────────────
# Plot R² with Early Stopping
# ────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(train_r2_hist, label="Train R²")
plt.plot(test_r2_hist, label="Test R²")
plt.axvline(best_epoch, color='red', linestyle='--', label=f"Early Stop @ {best_epoch}")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("Training vs Test R² with Early Stopping")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ────────────────────────────────
# Optional: Plot Fit Line (1D case)
# ────────────────────────────────
if x_test.shape[1] == 1:
    x_axis = x_test[:, 0]
    x_sorted_idx = np.argsort(x_axis)
    x_sorted = x_axis[x_sorted_idx]
    y_sorted = y_test[x_sorted_idx]
    y_pred_sorted = y_pred_test[x_sorted_idx]

    plt.figure(figsize=(8, 5))
    plt.scatter(x_sorted, y_sorted, label="Actual")
    plt.plot(x_sorted, y_pred_sorted, label="Predicted", color="red")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Model Prediction on Test Data")
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
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Value")
plt.ylabel("Residual (True - Predicted)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.show()
