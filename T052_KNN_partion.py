# ================================================================
#  KD-Tree visual demo: partitions, boundaries, and K-NN classifier
# ================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────
#  Helper – recursively draw KD-tree rectangles (2-D only)
# ─────────────────────────────────────────────────────────
def draw_kd_rects(ax, data, depth=0, bounds=None, leaf_size=5):
    """
    Visualise KD-tree partitions built with median splits.
    `data`     : ndarray (n_samples, 2)
    `bounds`   : [xmin, xmax, ymin, ymax]
    """
    if bounds is None:
        xmin, ymin = data.min(axis=0)
        xmax, ymax = data.max(axis=0)
        bounds = [xmin, xmax, ymin, ymax]

    # Stop when leaf reached
    if len(data) <= leaf_size:
        xmin, xmax, ymin, ymax = bounds
        ax.add_patch(Rectangle((xmin, ymin),
                               xmax - xmin, ymax - ymin,
                               fill=False, lw=1.2, ec="grey", alpha=0.6))
        return

    axis = depth % 2                           # 0 → x-axis, 1 → y-axis
    median = np.median(data[:, axis])
    left_mask = data[:, axis] <= median
    right_mask = ~left_mask

    if axis == 0:  # vertical split
        ax.axvline(median, ls="--", lw=1, c="k", alpha=0.7)
        draw_kd_rects(ax, data[left_mask], depth+1,
                      [bounds[0], median, bounds[2], bounds[3]], leaf_size)
        draw_kd_rects(ax, data[right_mask], depth+1,
                      [median, bounds[1], bounds[2], bounds[3]], leaf_size)
    else:          # horizontal split
        ax.axhline(median, ls="--", lw=1, c="k", alpha=0.7)
        draw_kd_rects(ax, data[left_mask], depth+1,
                      [bounds[0], bounds[1], bounds[2], median], leaf_size)
        draw_kd_rects(ax, data[right_mask], depth+1,
                      [bounds[0], bounds[1], median, bounds[3]], leaf_size)

# ─────────────────────────────────────────────────────────
#  1-D quick look – identical to the snippet you posted
# ─────────────────────────────────────────────────────────
def oned_demo():
    x = np.arange(-2.2, 2.2, 0.2)[:, None]       # shape (n, 1)
    y = (x[:, 0] >= 0).astype(np.int8)

    # Build KD-tree & draw vertical boundaries
    tree = KDTree(x, leaf_size=5)
    bounds = tree.node_bounds.base.reshape(-1)

    colors = ["dodgerblue", "crimson"]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    for i, xi in enumerate(x):
        ax.scatter(xi[0], 0, c=colors[y[i]], s=100, zorder=3, edgecolors="k")
    for b in bounds:
        ax.axvline(b, lw=1.5, c="grey", alpha=0.7)
    ax.set_title("1-D KD-tree partitions")
    ax.set_yticks([])
    plt.tight_layout()

# ─────────────────────────────────────────────────────────
#  2-D KD-tree partitions + K-NN decision boundary
# ─────────────────────────────────────────────────────────
def twod_demo():
    # Synthetic binary classification problem, 2 informative features
    X, y = make_classification(n_samples=500,
                               n_features=2,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Fit K-NN with KD-tree backend
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k, algorithm="kd_tree",
                               leaf_size=20, n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"Test accuracy (k={k}): {acc:.3f}")

    # Plot data points and KD-tree partitions
    fig, ax = plt.subplots(figsize=(7, 6))
    draw_kd_rects(ax, X_train, leaf_size=20)

    # Scatter training points
    ax.scatter(*X_train[y_train == 0].T, s=35, c="dodgerblue", label="class 0")
    ax.scatter(*X_train[y_train == 1].T, s=35, c="crimson",     label="class 1")

    # Overlay K-NN decision boundary on a fine grid
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 300),
                         np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = knn.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.15, levels=[-0.5, 0.5, 1.5],
                colors=["dodgerblue", "crimson"])
    ax.set_title("2-D KD-tree partitions (dashed) vs. K-NN decision region")
    ax.legend(loc="upper right")
    plt.tight_layout()

# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    oned_demo()
    twod_demo()
    plt.show()
