import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PARAMS = {
    "train_csv": "./data/energy-train-s.csv",
    "test_csv": "./data/energy-test-s.csv",
    "degree": 7,
    "alpha": 2.0,
    "reg_mode": "l1+l2",  # 'l2' | 'l1' | 'l1+l2'
    "elastic_r": 0.15,  #  reg_mode='l1+l2'
    "learning_rate": 1e-3,
    "epochs": 10_000,
    "patience": 5,
    "random_seed": 0,
    "draw_plots": True
}
np.random.seed(PARAMS["random_seed"])


train_df = pd.read_csv(PARAMS["train_csv"])
test_df = pd.read_csv(PARAMS["test_csv"])

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values.reshape(-1, 1)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values.reshape(-1, 1)

poly = PolynomialFeatures(PARAMS["degree"], include_bias=True)
Phi_train = poly.fit_transform(X_train)
Phi_test = poly.transform(X_test)



def reg_penalty(w):
    if PARAMS["reg_mode"] == "l2":
        return PARAMS["alpha"] * np.sum(w ** 2)
    if PARAMS["reg_mode"] == "l1":
        return PARAMS["alpha"] * np.sum(np.abs(w))
    # elastic-net
    l1 = np.sum(np.abs(w))
    l2 = np.sum(w ** 2)
    r = PARAMS["elastic_r"]
    return PARAMS["alpha"] * (r * l1 + 0.5 * (1 - r) * l2)


def reg_grad(w):
    if PARAMS["reg_mode"] == "l2":
        return 2 * PARAMS["alpha"] * w
    if PARAMS["reg_mode"] == "l1":
        return PARAMS["alpha"] * np.sign(w)
    r = PARAMS["elastic_r"]
    return PARAMS["alpha"] * (r * np.sign(w) + (1 - r) * w)



n_samples, n_features = Phi_train.shape
w = np.random.randn(n_features, 1) * 0.01

best_w = w.copy()
best_mse = float("inf")
wait = 0

train_mse_hist, test_mse_hist = [], []
train_r2_hist, test_r2_hist = [], []

for epoch in range(PARAMS["epochs"]):
    # پیش‌بینی و خطا
    y_hat_train = np.dot(Phi_train, w)
    error = y_hat_train - y_train

    # گرادیان کل = گرادیان خطای مربعی + گرادیان منظم‌سازی
    grad = (2 / n_samples) * np.dot(Phi_train.T, error) + reg_grad(w)
    w -= PARAMS["learning_rate"] * grad


    mse_train = mean_squared_error(y_train, y_hat_train) + reg_penalty(w)
    r2_train = r2_score(y_train, y_hat_train)


    y_hat_test = np.dot(Phi_test, w)
    mse_test = mean_squared_error(y_test, y_hat_test) + reg_penalty(w)
    r2_test = r2_score(y_test, y_hat_test)

    train_mse_hist.append(mse_train)
    test_mse_hist.append(mse_test)
    train_r2_hist.append(r2_train)
    test_r2_hist.append(r2_test)


    if mse_test < best_mse - 1e-9:
        best_mse = mse_test
        best_w = w.copy()
        wait = 0
    else:
        wait += 1
        if wait > PARAMS["patience"]:
            print(f"Early-stopped at epoch {epoch}  (best test MSE = {best_mse:.5f})")
            break


    if epoch % 100 == 0:
        print(f"Epoch {epoch:5d} | train MSE {mse_train:.4f} | test MSE {mse_test:.4f}")


y_pred = np.dot(Phi_test, best_w)
print("\nFinal test metrics")
print(" MAE :", mean_absolute_error(y_test, y_pred))
print(" MSE :", mean_squared_error(y_test, y_pred))
print(" R²  :", r2_score(y_test, y_pred))


if PARAMS["draw_plots"]:
    plt.figure(figsize=(8, 4))
    plt.plot(train_mse_hist, label="train MSE")
    plt.plot(test_mse_hist, label="test MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Mean-Squared Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8, 4))
    plt.plot(train_r2_hist, label="train R²")
    plt.plot(test_r2_hist, label="test R²")
    plt.xlabel("Epoch")
    plt.ylabel("R² score")
    plt.title("Coefficient of Determination")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    if X_train.shape[1] == 1:
        xx = np.linspace(X_train.min(), X_train.max(), 500).reshape(-1, 1)
        yy = poly.transform(xx).dot(best_w)
        plt.figure(figsize=(8, 5))
        plt.scatter(X_train, y_train, color="steelblue", alpha=0.6, label="train")
        plt.scatter(X_test, y_test, color="limegreen", alpha=0.6, label="test")
        plt.plot(xx, yy, color="crimson", linewidth=2,
                 label=f"fit (degree={PARAMS['degree']})")
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Polynomial Fit")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
