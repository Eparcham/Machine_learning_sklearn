import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDClassifier
from mlxtend.plotting import plot_decision_regions

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

plot_decision_regions(X_train,y_train,clf=model)
plt.show()





