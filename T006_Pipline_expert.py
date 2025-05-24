import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ────────────────────────────────────────
# Load Data
# ────────────────────────────────────────
train_df = pd.read_csv('./data/auto-train-multi.csv')
test_df = pd.read_csv('./data/auto-test-multi.csv')

x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# ────────────────────────────────────────
# Config
# ────────────────────────────────────────
degree = 3
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3)
}

# ────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────
results = []

for name, model in models.items():
    pipe = make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        model
    )
    pipe.fit(x_train, y_train)

    y_train_pred = pipe.predict(x_train)
    y_test_pred = pipe.predict(x_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    results.append({
        "Model": name,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse
    })

# ────────────────────────────────────────
# Show Results
# ────────────────────────────────────────
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="Test R²", ascending=False))
