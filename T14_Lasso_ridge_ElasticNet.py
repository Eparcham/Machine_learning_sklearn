import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------
PARAMS = {
    "train_csv":   "./data/energy-train-s.csv",
    "test_csv":    "./data/energy-test-s.csv",
    "degree":      5,            # polynomial degree
    "ridge_alpha": 25.0,          # L2 strength
    "lasso_alpha": 0.01,         # L1 strength  (keep small to avoid all-zero)
    "enet_alpha":  0.5,          # overall strength for elastic-net
    "l1_ratio":    0.15,         # elastic-net mixing (0=l2, 1=l1)
    "random_seed": 42,
    "draw_plots":  True
}
np.random.seed(PARAMS["random_seed"])

# ------------------------------------------------------------------
# load data
# ------------------------------------------------------------------
train_df = pd.read_csv(PARAMS["train_csv"])
test_df  = pd.read_csv(PARAMS["test_csv"])

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:, -1].values

# ------------------------------------------------------------------
# build pipelines
# ------------------------------------------------------------------
models = {
    "Ridge": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(PARAMS["degree"], include_bias=False),
        Ridge(alpha=PARAMS["ridge_alpha"],
              random_state=PARAMS["random_seed"])
    ),
    "Lasso": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(PARAMS["degree"], include_bias=False),
        Lasso(alpha=PARAMS["lasso_alpha"],
              max_iter=30_000,                # plenty of iterations for convergence
              random_state=PARAMS["random_seed"])
    ),
    "ElasticNet": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(PARAMS["degree"], include_bias=False),
        ElasticNet(alpha=PARAMS["enet_alpha"],
                   l1_ratio=PARAMS["l1_ratio"],
                   max_iter=30_000,
                   random_state=PARAMS["random_seed"])
    )
}

# ------------------------------------------------------------------
# fit, predict, evaluate
# ------------------------------------------------------------------
preds, metrics = {}, {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    preds[name] = y_pred
    metrics[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R²":  r2_score(y_test, y_pred)
    }

# report
for name, m in metrics.items():
    print(f"{name:<10}  MAE {m['MAE']:.4f}   MSE {m['MSE']:.4f}   R² {m['R²']:.4f}")

# ------------------------------------------------------------------
# plots
# ------------------------------------------------------------------
if PARAMS["draw_plots"]:

    # 1) predicted vs true (scatter)
    plt.figure(figsize=(6,6))
    colors = {"Ridge":"tab:blue", "Lasso":"tab:orange", "ElasticNet":"tab:green"}
    for name, y_pred in preds.items():
        plt.scatter(y_test, y_pred, alpha=0.5, label=name, color=colors[name])
    lims = [y_test.min(), y_test.max()]
    plt.plot(lims, lims, "--k", linewidth=1)
    plt.xlabel("True value"); plt.ylabel("Predicted value")
    plt.title("Predicted vs true – three regularisers")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 2) “curve” view (samples sorted by target)
    order = np.argsort(y_test)
    plt.figure(figsize=(10,4))
    plt.plot(y_test[order], label="True", color="black", linewidth=2)
    for name, y_pred in preds.items():
        plt.plot(y_pred[order], label=name, color=colors[name])
    plt.xlabel("Sample index (sorted)"); plt.ylabel("Target")
    plt.title("Model output curves")
    plt.legend(); plt.grid(True, ls=":"); plt.tight_layout(); plt.show()
