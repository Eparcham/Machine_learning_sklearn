import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_aug  = np.hstack([np.ones((X_test.shape[0], 1)),  X_test])

Y_train = label_binarize(y_train, classes=[0, 1, 2])
Y_test  = label_binarize(y_test,  classes=[0, 1, 2])

# ── Model helpers ─────────────────────────────────────────────────────────────
def sigmoid(z): return 1 / (1 + np.exp(-z))

def forward(X, W): return sigmoid(X @ W)            # independent-sigmoid version
# def forward(X, W): return softmax(X @ W)          # soft-max version

def bce(Y, P, eps=1e-10):                           # use cce for soft-max
    return -np.mean(Y * np.log(P + eps) + (1-Y) * np.log(1-P + eps))

def grad(X, Y, P): return X.T @ (P - Y) / len(Y)

def accuracy(Y_true, P):
    y_true = np.argmax(Y_true, axis=1)
    y_pred = np.argmax(P,       axis=1)
    return np.mean(y_true == y_pred)

# ── Training loop ─────────────────────────────────────────────────────────────
np.random.seed(0)
W = np.random.randn(X_train_aug.shape[1], 3) * 0.01
lr = 0.1
epochs = 10_000
train_loss, test_loss = [], []

for epoch in range(epochs):
    P_train = forward(X_train_aug, W)
    W -= lr * grad(X_train_aug, Y_train, P_train)

    P_test = forward(X_test_aug, W)

    train_loss.append(bce(Y_train, P_train))
    test_loss.append(bce(Y_test,  P_test))

    if epoch % 1000 == 0:
        print(f"[{epoch:4d}] "
              f"train loss={train_loss[-1]:.4f} "
              f"test loss={test_loss[-1]:.4f} "
              f"train acc={accuracy(Y_train, P_train):.3f} "
              f"test acc={accuracy(Y_test,  P_test):.3f}")

# ── Final metrics ─────────────────────────────────────────────────────────────
print(f"\nFinal test accuracy: {accuracy(Y_test, forward(X_test_aug, W)):.3f}")

# ── Loss curve ────────────────────────────────────────────────────────────────
plt.plot(train_loss, label="train"); plt.plot(test_loss, label="test")
plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.legend(); plt.tight_layout()
plt.show()
