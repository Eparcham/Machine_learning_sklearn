import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, roc_curve

# ──────────────────────────────────────────────
# 1. Logistic Regression on External Dataset
# ──────────────────────────────────────────────
data = pd.read_csv("./data/exam.csv")
X = data['study_hours'].values.reshape(-1, 1)
y = data['pass_fail'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(penalty='l2')
model.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, clf=model)
plt.title("Decision Boundary - Exam Dataset")
plt.show()

# ──────────────────────────────────────────────
# 2. Thresholding Function
# ──────────────────────────────────────────────
def apply_threshold(y_prob, threshold=0.6):
    return np.where(y_prob < threshold, 0, 1)

# ──────────────────────────────────────────────
# 3. Logistic Regression on Synthetic Dataset
# ──────────────────────────────────────────────
X, y = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_classes=2, n_clusters_per_class=1, class_sep=1.0, random_state=5
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(penalty='l2')
model.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, clf=model)
plt.title("Decision Boundary - Synthetic Dataset")
plt.show()

# ──────────────────────────────────────────────
# 4. Prediction Probability and Histogram
# ──────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]

neg = y_prob[y_test == 0]
pos = y_prob[y_test == 1]

plt.hist(neg, bins=75, alpha=0.6, label='Class 0')
plt.hist(pos, bins=75, alpha=0.6, label='Class 1')
plt.title("Probability Distribution")
plt.legend()
plt.show()

# ──────────────────────────────────────────────
# 5. Metric Calculation Function
# ──────────────────────────────────────────────
def evaluate_metrics(y_pred, y_true):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total != 0 else 0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0

    return tpr, tnr, fpr, fnr, acc, tp, tn, fp, fn

# ──────────────────────────────────────────────
# 6. Custom ROC Curve Plotting
# ──────────────────────────────────────────────
thresholds = np.linspace(0, 1, 100)
TPR = []
FPR = []

for t in thresholds:
    y_pred = apply_threshold(y_prob, threshold=t)
    tpr, tnr, fpr, fnr, acc, tp, tn, fp, fn = evaluate_metrics(y_pred, y_test)
    TPR.append(tpr)
    FPR.append(fpr)

plt.plot(FPR, TPR, label='Custom ROC', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Manual)')
plt.legend()
plt.grid(True)
plt.show()

# ──────────────────────────────────────────────
# 7. Built-in ROC Curve
# ──────────────────────────────────────────────
fpr, tpr, threshold_values = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='Sklearn ROC', color='green')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Sklearn)')
plt.legend()
plt.grid(True)
plt.show()
