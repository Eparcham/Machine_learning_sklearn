import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.neighbors import KNeighborsClassifier, KDTree, NearestNeighbors
from scipy.stats import mode
from sympy.abc import alpha

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
    random_state=10
)


plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
plt.title("Training Data")
plt.show()

kdt = KDTree(x,leaf_size=10)
query_point = np.array([[0,0]])

# _,ind = kdt.query(query_point, k=3)



n = 200
xmin, xmax = x.min(0)-0.5, x.max(0)+0.5
x1r = np.linspace(xmin[0], xmax[0], n)
x2r = np.linspace(xmin[1], xmax[1], n)
x1m, x2m = np.meshgrid(x1r, x2r)
xm = np.stack((x1m.flatten(), x2m.flatten()), axis=1)
_,ind = kdt.query(xm,5)
y_hat = mode(y[ind],axis=1,keepdims=False).mode
y_hat = y_hat.reshape(x1m.shape)
plt.contourf(x1m, x2m, y_hat, alpha=0.5, cmap='jet')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet',alpha=0.8)
plt.show()


