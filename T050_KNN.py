# =============================================================
#  Fast K-NN (Euclidean, Mahalanobis, Manhattan) + comparison plot
# =============================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from typing import Literal


# ─────────────────────────────────────────────────────────
#  Custom K-NN classifier
# ─────────────────────────────────────────────────────────
class KNNClassifier:
    """
    k-nearest-neighbour classifier supporting
        • Euclidean      ('euclidean')
        • Mahalanobis    ('mahalanobis')
        • Manhattan/ℓ₁   ('cityblock')
    """
    Metric = Literal["euclidean", "mahalanobis", "cityblock"]

    def __init__(self, k: int = 3, metric: Metric = "euclidean"):
        if metric not in {"euclidean", "mahalanobis", "cityblock"}:
            raise ValueError("metric must be 'euclidean', 'mahalanobis', or 'cityblock'")
        self.k = k
        self.metric = metric
        self.VI = None  # inverse covariance for Mahalanobis

    # ---------------------- training ----------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.ascontiguousarray(X)
        self.y_train = np.asarray(y)

        if self.metric == "mahalanobis":
            cov = np.cov(self.X_train, rowvar=False)
            cov += 1e-10 * np.eye(cov.shape[0])  # regularisation
            self.VI = np.linalg.inv(cov)
        return self

    # ---------------- distance matrix ---------------------
    def _distance_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "mahalanobis":
            return cdist(X, self.X_train, metric="mahalanobis", VI=self.VI)
        return cdist(X, self.X_train, metric=self.metric)

    # --------------------- prediction ---------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        D = self._distance_matrix(X)
        idx = np.argpartition(D, self.k, axis=1)[:, : self.k]
        labels = self.y_train[idx]
        return mode(labels, axis=1, keepdims=False).mode.astype(self.y_train.dtype)

    # ------------------------ score -----------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# ─────────────────────────────────────────────────────────
#  Demo: compare metrics on synthetic 2-D data
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate a 2-D binary classification task
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.5,
        random_state=10,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    k = 3
    clfs = {
        "Euclidean":   KNNClassifier(k, metric="euclidean").fit(X_train, y_train),
        "Mahalanobis": KNNClassifier(k, metric="mahalanobis").fit(X_train, y_train),
        "Manhattan":   KNNClassifier(k, metric="cityblock").fit(X_train, y_train),
    }

    # Collect accuracies
    scores = {name: clf.score(X_test, y_test) for name, clf in clfs.items()}
    for name, acc in scores.items():
        print(f"{name:11s}: {acc:.3f}")

    # Plot comparison
    plt.figure(figsize=(5, 4))
    plt.bar(scores.keys(), scores.values(), width=0.5)
    plt.ylabel("Accuracy")
    plt.title(f"K-NN (k={k}) — metric comparison")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
