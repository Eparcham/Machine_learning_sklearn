import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.svm import SVC,LinearSVC, NuSVC
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

svm = LinearSVC(loss='hinge',C=0.1, max_iter= 2000)
svm.fit(x_train, y_train)

def draw_plt(x_test,y_test, model):
    n = 1000
    xmin, xmax = x_test.min(0), x_test.max(0)
    x1r = np.linspace(xmin[0], xmax[0], n)
    x2r = np.linspace(xmin[1], xmax[1], n)
    x1m, x2m = np.meshgrid(x1r, x2r)
    xm = np.stack((x1m.flatten(), x2m.flatten()), axis=1)
    y_hat = model.decision_function(xm)
    y_hat = y_hat.reshape(x1m.shape)
    plt.contour(x1m, x2m, y_hat, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='jet',s=50,zorder=3)
    plt.show()


draw_plt(x_test,y_test, svm)
draw_plt(x_train,y_train, svm)

svc = SVC(kernel='linear',random_state=0)
svc.fit(x_train, y_train)
draw_plt(x_test,y_test, svc)
draw_plt(x_train,y_train, svc)


model = NuSVC(nu=0.01, kernel='linear',random_state=0, probability=True)
model.fit(x_train, y_train)
draw_plt(x_test,y_test, model)
draw_plt(x_train,y_train, model)