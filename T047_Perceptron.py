from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, SGDClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ──────────────────────────────────────────────
# Dataset Preparation
# ──────────────────────────────────────────────
x, y = make_classification(
    n_samples=500,
    n_features=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=5
)
y = np.where(y == 0, -1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis')
plt.title("Training Data")
plt.show()

# ──────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────
def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def bce(y, y_hat):
    return np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def accuracy(y, y_hat, threshold=0.0):
    y_pred = np.where(y_hat < threshold, -1, 1)
    return np.mean(y_pred == y)

# ──────────────────────────────────────────────
# Custom Perceptron Implementation
# ──────────────────────────────────────────────
class Perceptron:
    def __init__(self, in_features=2, maxit=2000, eta=0.09, w=None, b=None):
        self.w = w if w is not None else np.random.rand(in_features, 1)
        self.b = b if b is not None else np.random.rand()
        self.maxit = maxit
        self.eta = eta
        self.loss_history = []

    def predict(self, x):
        return x @ self.w + self.b

    def loss_fn(self, y, y_hat):
        return np.mean(np.maximum(0, -y * y_hat))

    def grad(self, x, y, y_hat):
        grad_w = (-y * x * np.heaviside(-y * y_hat, 1)).mean(axis=0).reshape(self.w.shape)
        grad_b = (-y * np.heaviside(-y * y_hat, 1)).mean(axis=0)
        return grad_w, grad_b

    def fit(self, x, y):
        y = y.reshape(-1, 1)  # Ensure shape is (N, 1)
        for i in range(self.maxit):
            y_hat = self.predict(x)
            loss = self.loss_fn(y, y_hat)
            self.loss_history.append(loss)

            grad_w, grad_b = self.grad(x, y, y_hat)
            self.w -= self.eta * grad_w
            self.b -= self.eta * grad_b

            acc = accuracy(y, y_hat, threshold=0.0)
            if i % 10 == 0:
                print(f"Epoch = {i}, Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    def score(self, x, y):
        y_hat = self.predict(x)
        y = y.reshape(-1, 1)  # ensure 2D
        return accuracy(y, y_hat, threshold=0.0)


# ──────────────────────────────────────────────
# Wrapper for Decision Region Plotting
# ──────────────────────────────────────────────
class PerceptronWrapper:
    def __init__(self, w=None, b=None):
        self.w = w
        self.b = b

    def predict(self, x):
        y_hat = x @ self.w + self.b
        return np.where(y_hat > 0, 1, 0)  # for mlxtend (expects 0/1 labels)

# ──────────────────────────────────────────────
# Train and Visualize
# ──────────────────────────────────────────────
model = Perceptron()
model.fit(x_train, y_train[:, None])

wrapper = PerceptronWrapper(w=model.w, b=model.b)
plot_decision_regions(x_test, y_test, clf=wrapper)
plt.title("Perceptron Decision Boundary")
plt.show()

test_acc = model.score(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
