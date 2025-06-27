#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import mode

class WeightedKNN:
    """
    k-nearest neighbours with inverse-distance weighting.
    metrics: 'euclidean', 'mahalanobis', 'cityblock' (Manhattan), 'cosine'
    """
    def __init__(self, k=3, metric="euclidean"):
        if metric not in {"euclidean", "mahalanobis", "cityblock", "cosine"}:
            raise ValueError("invalid metric")
        self.k = k
        self.metric = metric
        self.VI = None   # inverse covariance (Mahalanobis)

    def fit(self, X, y):
        self.X = np.ascontiguousarray(X)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        if self.metric == "mahalanobis":
            cov = np.cov(self.X, rowvar=False) + 1e-10 * np.eye(X.shape[1])
            self.VI = np.linalg.inv(cov)
        return self

    # ---------- helpers ----------
    def _dmatrix(self, X):
        if self.metric == "mahalanobis":
            return cdist(X, self.X, metric="mahalanobis", VI=self.VI)
        return cdist(X, self.X, metric=self.metric)

    def _k_neigh(self, D):
        idx = np.argpartition(D, self.k, axis=1)[:, : self.k]
        dist = D[np.arange(D.shape[0])[:, None], idx]
        lbls = self.y[idx]
        return dist, lbls

    # ---------- public API ----------
    def predict_proba(self, X, eps=1e-9):
        D = self._dmatrix(X)
        dist_k, lbls_k = self._k_neigh(D)
        w = 1.0 / (dist_k + eps)                       # inverse-distance weights
        probs = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            mask = (lbls_k == c)
            probs[:, i] = np.sum(w * mask, axis=1)
        probs /= probs.sum(axis=1, keepdims=True)      # normalise
        return probs

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ───────── demo ─────────
if __name__ == "__main__":
    X, y = make_classification(
        n_samples=500, n_features=2, n_redundant=0,
        n_clusters_per_class=1, class_sep=1.0, random_state=10
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = WeightedKNN(k=5, metric="euclidean").fit(Xtr, ytr)
    acc = knn.score(Xte, yte)
    probs = knn.predict_proba(Xte)          # shape (n_test, n_classes)

    print(f"accuracy: {acc:.3f}")
    print("first 5 predictions and probabilities:")
    for i in range(5):
        pred = knn.predict(Xte[i:i+1])[0]
        p0, p1 = probs[i]
        print(f"sample {i}: pred={pred}, P(class0)={p0:.3f}, P(class1)={p1:.3f}")
