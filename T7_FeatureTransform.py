import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer,
    FunctionTransformer,
    StandardScaler
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
train_df = pd.read_csv('./data/auto-train-preprocessed.csv')
test_df = pd.read_csv('./data/auto-test-preprocessed.csv')

x_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values
x_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Define transformations
transformations = {
    "Polynomial(degree=3)": PolynomialFeatures(degree=3, include_bias=False),
    "PowerTransformer": PowerTransformer(),  # "yeo-johnson" by default
    "QuantileTransformer": QuantileTransformer(n_quantiles=6, output_distribution="normal", random_state=42),
    "SplineTransformer": SplineTransformer(n_knots=20, degree=3, include_bias=False),
    "Log(x+1)": FunctionTransformer(lambda x: np.log1p(np.abs(x))),  # Logarithmic transform
}

# Model
base_model = LinearRegression()

# Evaluation results
results = []

# Run experiments
for name, transformer in transformations.items():
    pipe = make_pipeline(
        StandardScaler(),  # normalize before transformation (optional but recommended)
        transformer,
        base_model
    )

    pipe.fit(x_train, y_train)
    y_train_pred = pipe.predict(x_train)
    y_test_pred = pipe.predict(x_test)

    results.append({
        "Transform": name,
        "Train R²": r2_score(y_train, y_train_pred),
        "Test R²": r2_score(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    })

# Show results
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="Test R²", ascending=False))

# Visualization (example for PowerTransformer)
pt = PowerTransformer()
xt_ = pt.fit_transform(x_train)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(x_train.flatten(), bins=50)
plt.title("Original Features")

plt.subplot(1, 2, 2)
plt.hist(xt_.flatten(), bins=50)
plt.title("Power Transformed Features")

plt.tight_layout()
plt.show()
