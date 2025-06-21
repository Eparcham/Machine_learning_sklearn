import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,SplineTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, SGDClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve


data = pd.read_csv("./data/exam.csv")

x=data['study_hours'].values
y=data['pass_fail'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = LogisticRegression(penalty='l2')

model.fit(x_train.reshape(-1, 1),y_train)
plot_decision_regions(x_train.reshape(-1, 1),y_train,clf=model)
plt.show()

## threshold
y_hat = model.predict_proba(x_test.reshape(-1, 1))[:,1]
def thr(y_hat,t=0.6):
    # y_hat[y_hat<t] = 0
    # y_hat[y_hat>=t] = 1
    y_hat_binary = np.where(y_hat<t,0,1)
    return y_hat_binary

# y_hat_binary = thr(y_hat,t=0.2)

if 1:
    x,y = make_classification(n_samples=500,n_features=2,random_state=5, n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=1.0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LogisticRegression(penalty='l2')

    model.fit(x_train,y_train)
    plot_decision_regions(x_train,y_train,clf=model)
    plt.show()


    y_hat = model.predict_proba(x_test)[:,1]
    negative = y_hat[y_test==0]
    positive = y_hat[y_test==1]

    plt.hist(negative,75)
    plt.hist(positive,75)
    plt.show()

def info(y_hat_binary, y_test):
    tp = np.sum((y_hat_binary==1) & (y_test==1))
    tn = np.sum((y_hat_binary==0) & (y_test==0))
    fp = np.sum((y_hat_binary==1) & (y_test==0))
    fn = np.sum((y_hat_binary==0) & (y_test==1))

    all = tp + tn + fp + fn
    acc = (tp+tn)/(tp+tn+fp+fn)
    rp = tp + fn
    pp = tp + fp
    rn = tn + fp
    pn = fn + tn

    if rp!=0:
       tpr = tp / (tp + fn)
       fnr = fn / (tp + fn)
    else:
        tpr = 0
        fnr = 0
    if rn!=0:
        tnr = tn/(tn + fp)
        fpr = fp/(tn + fp)
    else:
        tnr = 0
        fpr = 0

    return tpr,tnr,fpr,fnr,acc, tp, tn, fp, fn


# tpr,tnr,fpr,fnr,acc, tp, tn, fp, fn = info(y_hat_binary, y_test)
# print(confusion_matrix(y_test,y_hat_binary))
#
# print(f"tp:{tp}, tn:{tn}, fp:{fp}, fn:{fn} ,acc:{acc}")
# print(f"tpr:{tpr}, tnr:{tnr}, fpr:{fpr}, fnr:{fnr}")

## roc curve

th = np.linspace(0,1,100)
y_hat = model.predict_proba(x_test)[:,1]
TPR = []
FPR = []
for t in th:
    y_hat_binary = thr(y_hat,t)
    tpr, tnr, fpr, fnr, acc, tp, tn, fp, fn = info(y_hat_binary, y_test)
    tn,fp,fn,tp = confusion_matrix(y_test, y_hat_binary).ravel()
    tpr1 = tp/(tp+fn)
    fpr1 = fp/(tn+fp)
    TPR.append(tpr)
    FPR.append(fpr)

tpra = np.array(TPR)
fpra = np.array(FPR)

plt.plot(fpra,tpra,label='roc')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


fpr,tpr,th = roc_curve(y_test,y_hat)
plt.plot(fpr,tpr,label='roc')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
