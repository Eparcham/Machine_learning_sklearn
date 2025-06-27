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
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=3
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
class SVM:
    def __init__(self, in_features=2, maxit=2000, eta=0.09, w=None, b=None, c=0.5):
        self.w = w if w is not None else np.random.rand(in_features, 1)
        self.b = b if b is not None else np.random.rand()
        self.c = c
        self.maxit = maxit
        self.eta = eta
        self.loss_history = []

    def predict(self, x):
        return x @ self.w + self.b

    def loss_fn(self, y, y_hat,mask):
        y_mask = y[mask]
        y_hat_mask = y_hat[mask]
        return np.mean(np.maximum(0,1-y_mask * y_hat_mask))

    def grad(self, x, y, y_hat,mask):
        x_mask = x[mask]
        y_mask = y[mask]
        # y_hat_mask = y_hat[mask]
        grad_w = (-y_mask * x_mask).mean(axis=0).reshape(self.w.shape) + self.c * self.w
        grad_b = (-y_mask ).mean(axis=0)
        return grad_w, grad_b

    def fit(self, x, y):
        y = y.reshape(-1, 1)  # Ensure shape is (N, 1)
        for i in range(self.maxit):
            y_hat = self.predict(x)
            mask = np.squeeze((1-y_hat*y)>0)
            # print(mask.sum())
            if mask.sum()==0:
                print(i, "break!")
                break
            loss = self.loss_fn(y, y_hat, mask)
            self.loss_history.append(loss)

            grad_w, grad_b = self.grad(x, y, y_hat,mask)
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
class SVMWrapper:
    def __init__(self, w=None, b=None):
        self.w = w
        self.b = b

    def predict(self, x):
        y_hat = x @ self.w + self.b
        return np.where(y_hat > 0, 1, 0)  # for mlxtend (expects 0/1 labels)

    def predict_me(self, x):
        y_hat = x @ self.w + self.b
        return y_hat  # for mlxtend (expects 0/1 labels)

# ──────────────────────────────────────────────
# Train and Visualize
# ──────────────────────────────────────────────
model = SVM()
model.fit(x_train, y_train[:, None])

wrapper = SVMWrapper(w=model.w, b=model.b)
plot_decision_regions(x_test, y_test, clf=wrapper)
plt.title("Perceptron Decision Boundary")
plt.show()

test_acc = model.score(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

def draw_plt(x_test,y_test, model):
    n = 1000
    xmin, xmax = x_test.min(0), x_test.max(0)
    x1r = np.linspace(xmin[0], xmax[0], n)
    x2r = np.linspace(xmin[1], xmax[1], n)
    x1m, x2m = np.meshgrid(x1r, x2r)
    xm = np.stack((x1m.flatten(), x2m.flatten()), axis=1)
    y_hat = model.predict(xm)
    y_hat = y_hat.reshape(x1m.shape)
    plt.contour(x1m, x2m, y_hat, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='jet',s=50,zorder=3)
    plt.show()


draw_plt(x_test,y_test, model)
draw_plt(x_train,y_train, model)
