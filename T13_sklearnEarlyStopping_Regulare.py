#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------
CFG = {
    "train_csv":      "./data/energy-train-s.csv",
    "test_csv":       "./data/energy-test-s.csv",
    "degree":         5,
    "eta0":           1e-4,
    "max_epochs":     10_000,
    "patience":       8,         # early-stopping patience
    "random_seed":    42,
    "alpha":          0.5,       # for elastic-net model
    "l1_ratio":       0.15
}
np.random.seed(CFG["random_seed"])

# ------------------------------------------------------------------
# load data
# ------------------------------------------------------------------
train_df = pd.read_csv(CFG["train_csv"])
test_df  = pd.read_csv(CFG["test_csv"])

X_train_full = train_df.iloc[:, :-1].values
y_train_full = train_df.iloc[:, -1].values
X_test       = test_df.iloc[:, :-1].values
y_test       = test_df.iloc[:, -1].values

# split train→train/val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=CFG["random_seed"]
)

# polynomial expansion + scaling
poly = PolynomialFeatures(CFG["degree"], include_bias=False)
scaler = StandardScaler()

X_tr_poly  = scaler.fit_transform(poly.fit_transform(X_tr))
X_val_poly = scaler.transform(poly.transform(X_val))
X_test_poly = scaler.transform(poly.transform(X_test))

# ------------------------------------------------------------------
# helper to train SGD with manual loop
# ------------------------------------------------------------------
def train_sgd(Xtr, ytr, Xval, yval, penalty, alpha, l1_ratio, eta0,max_epochs, patience):
    n_features = Xtr.shape[1]
    model = SGDRegressor(loss="squared_error",
                         penalty=penalty,
                         alpha=alpha,
                         l1_ratio=l1_ratio,
                         learning_rate="constant",
                         eta0=eta0,
                         max_iter=1,      # we call partial_fit manually
                         shuffle=True,
                         tol=None,
                         random_state=CFG["random_seed"],
                         warm_start=True)

    # initialise once
    model.partial_fit(Xtr, ytr)

    train_mse, val_mse = [], []
    best_val = float("inf")
    best_coef = model.coef_.copy()
    wait = 0

    for epoch in range(max_epochs):
        model.partial_fit(Xtr, ytr)

        ytr_pred  = model.predict(Xtr)
        yval_pred = model.predict(Xval)
        mse_tr  = mean_squared_error(ytr,  ytr_pred)
        mse_val = mean_squared_error(yval, yval_pred)

        train_mse.append(mse_tr)
        val_mse.append(mse_val)

        if mse_val < best_val - 1e-10:
            best_val = mse_val
            best_coef = model.coef_.copy()
            wait = 0
        else:
            wait += 1
            if wait > patience:
                print(f"[{penalty or 'none':9s}] early stop at epoch {epoch:5d}   "
                      f"best val MSE {best_val:.4f}")
                break

    model.coef_ = best_coef  # restore best weights
    return model, train_mse, val_mse

# ------------------------------------------------------------------
# train two models
# ------------------------------------------------------------------
model_none, tr_none, val_none = train_sgd(
    X_tr_poly, y_tr, X_val_poly, y_val,
    penalty=None, alpha=0.0, l1_ratio=0.0,
    eta0=CFG["eta0"], max_epochs=CFG["max_epochs"], patience=CFG["patience"]
)

model_elastic, tr_el, val_el = train_sgd(
    X_tr_poly, y_tr, X_val_poly, y_val,
    penalty="elasticnet", alpha=CFG["alpha"], l1_ratio=CFG["l1_ratio"],
    eta0=CFG["eta0"], max_epochs=CFG["max_epochs"], patience=CFG["patience"]
)

# ------------------------------------------------------------------
# evaluation on test set
# ------------------------------------------------------------------
def report(name, mdl):
    yp = mdl.predict(X_test_poly)
    print(f"\n{name}  |  Test MAE {mean_absolute_error(y_test, yp):.4f}  "
          f"MSE {mean_squared_error(y_test, yp):.4f}  R² {r2_score(y_test, yp):.4f}")
    return yp
y_pred_none    = report("No-reg     ", model_none)
y_pred_elastic = report("Elastic-net", model_elastic)

# ------------------------------------------------------------------
# 1) learning curves
# ------------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(tr_none,   label="train (none)",  color="tab:blue",  linewidth=1)
plt.plot(val_none,  label="val   (none)",  color="tab:blue",  linestyle="--")
plt.plot(tr_el,     label="train (elastic)", color="tab:orange", linewidth=1)
plt.plot(val_el,    label="val   (elastic)", color="tab:orange", linestyle="--")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("Learning curves")
plt.legend(); plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 2) coefficient magnitude comparison
# ------------------------------------------------------------------
coef_idx = np.arange(len(model_none.coef_))
plt.figure(figsize=(9,4))
plt.stem(coef_idx, np.abs(model_none.coef_), linefmt="tab:blue", markerfmt=" ", basefmt=" ")
plt.stem(coef_idx, np.abs(model_elastic.coef_), linefmt="tab:orange", markerfmt=" ", basefmt=" ")
plt.yscale("log")
plt.xlabel("Feature index")
plt.ylabel("|coef| (log scale)")
plt.title("Coefficient magnitudes\n(blue: no regularisation  |  orange: elastic-net)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3) residual plot for elastic-net model
# ------------------------------------------------------------------
residuals = y_test - y_pred_elastic
plt.figure(figsize=(6,5))
plt.scatter(y_pred_elastic, residuals, alpha=0.6)
plt.axhline(0, color="red", ls="--")
plt.xlabel("Predicted value")
plt.ylabel("Residual")
plt.title("Residuals (elastic-net)")
plt.grid(True); plt.tight_layout(); plt.show()


# ------------------------------------------------------------------
# 4) CURVE-LIKE COMPARISONS ON THE TEST SET
# ------------------------------------------------------------------

# (a) samples sorted by the true target – gives a “curve” for each model
order = np.argsort(y_test)
plt.figure(figsize=(10,4))
plt.plot(y_test[order],              label="True target",    color="black", linewidth=2)
plt.plot(y_pred_none[order],         label="No regulariser", color="tab:blue")
plt.plot(y_pred_elastic[order],      label="Elastic-net",    color="tab:orange")
plt.xlabel("Sample index (sorted by true value)")
plt.ylabel("Target value")
plt.title("Model output curves on test set")
plt.legend(); plt.grid(True, ls=":")
plt.tight_layout()
plt.show()

# (b) scatter plot with y = x reference line
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_none,    alpha=0.5, label="No regulariser", color="tab:blue")
plt.scatter(y_test, y_pred_elastic, alpha=0.5, label="Elastic-net",    color="tab:orange")
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "--k", lw=1)   # 45-degree reference
plt.xlabel("True value"); plt.ylabel("Predicted value")
plt.title("Predicted vs. true – two models")
plt.legend(); plt.grid(True, ls=":")
plt.tight_layout()
plt.show()
