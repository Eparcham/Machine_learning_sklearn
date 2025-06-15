import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler,SplineTransformer, PolynomialFeatures
from sklearn.datasets import load_breast_cancer,make_classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.pipeline import make_pipeline


data = np.loadtxt("./data/ex2data1.txt", delimiter=",")

x=data[:,:-1].copy()
y=data[:,-1].copy().astype(np.int64)

normz = StandardScaler()
X=normz.fit_transform(x)

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


model = make_pipeline(PolynomialFeatures(degree=2,interaction_only=True),LogisticRegression())

model.fit(X,y)
plot_decision_regions(X,y,clf=model)
plt.show()
