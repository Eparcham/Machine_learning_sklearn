import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer,make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from mlxtend.plotting import plot_decision_regions


x,y = make_classification(n_samples=500,n_features=2,random_state=5, n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=2)

plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

modle = LogisticRegression()
modle.fit(x,y)

## boundry desion

x1_min,x2_min = x.min(0)
x1_max,x2_max = x.max(0)

n = 500
x1r= np.linspace(x1_min,x1_max,n)
x2r= np.linspace(x2_min,x2_max,n)

x1m,x2m = np.meshgrid(x1r,x2r)

Xm = np.stack([x1m.ravel(),x2m.ravel()],axis=-1)
ym = modle.decision_function(Xm)


plt.scatter(x[:,0],x[:,1],c=y)
# plt.contour(x1m,x2m,ym.reshape(x1m.shape), levels=[0],)
plt.contour(x1m,x2m,ym.reshape(x1m.shape), levels=[-1, 0 , 1],)
plt.show()

## mlxtend
plot_decision_regions(x,y, clf=modle)
plt.show()

