import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, SGDClassifier

# ────────────────────────────────
# Load and Prepare Data
# ────────────────────────────────
x, y = load_breast_cancer(return_X_y=True)

# Correct order of split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=3)

# Normalize features
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=10000,solver='sag',random_state=3)
model.fit(x_train, y_train)

# Predictions
y_proba = model.predict_proba(x_test)
y_pred = model.predict(x_test)

# Evaluation
print("Accuracy on test set:", model.score(x_test, y_test))

model = SGDClassifier(loss = 'log_loss',random_state=27)
model.fit(x_train, y_train)
model.score(x_test, y_test)
print("Accuracy on train set:", model.score(x_train, y_train))
