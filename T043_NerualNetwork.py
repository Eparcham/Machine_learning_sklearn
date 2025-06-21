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


x,y = make_classification(n_samples=500,n_features=2,random_state=5, n_redundant=0,n_classes=2,n_clusters_per_class=1,class_sep=1.0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

##oop style
def relu(x):
    return np.maximum(x, 0)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def tanh(x):
    return np.tanh(x)
def bce(y,y_hat):
    return np.mean(-(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)))
def mse(y,y_hat):
    return np.mean((y-y_hat)**2)
def accuracy(y,y_hat,t=0.5):
    y_hat=np.where(y_hat>=t,1,0)
    acc = np.sum(y_hat==y)/len(y)
    return acc

class Neuron:
    def __init__(self,in_features,out_features,af=None,loss_fn=mse,maxit=200,eta=0.01):
        self.in_features=in_features
        self.out_features=out_features
        self.af=af
        self.loss_fn=loss_fn
        self.maxit = maxit
        self.losst_history = []
        self.eta = eta
        # weight and bias
        self.w = np.random.randn(self.in_features, self.out_features)
        self.b = np.random.randn(self.out_features)
        self.w_grad = None
        self.b_grad = None

    def fit(self,x,y):
        for i in range(self.maxit):
            y_hat = self.predict(x)
            loss = self.loss_fn(y,y_hat)
            self.losst_history.append(loss)
            self.gradient(x,y,y_hat)
            self.gradient_descent()
            acc = accuracy(y,y_hat,t=0.5)
            if i%10==0:
                print(f"epoch= {i}, loss= {loss} acc= {acc}")

    def gradient(self,x,y,y_hat):
        self.w_grad = (x.T@(y_hat-y))/len(y)
        self.b_grad = (y_hat-y).mean()

    def gradient_descent(self):
        self.w -= self.eta * self.w_grad
        self.b -= self.eta * self.b_grad

    def predict(self,x):
        y_hat= x@self.w+self.b
        y_hat = y_hat if self.af is None else self.af(y_hat)
        return y_hat
    def parameters(self):
        return {'w':self.w, 'b':self.b}

    def score(self,x,y):
        acc = accuracy(y,self.predict(x),t=0.5)
        return acc

Neuron_class = Neuron(in_features=2,out_features=1,af=sigmoid,loss_fn=bce,maxit=200,eta=0.01)
# print(Neuron_class.predict(x_train))
print(Neuron_class.parameters())
Neuron_class.fit(x_train,y_train[:,None])
print(Neuron_class.score(x_test,y_test[:,None]))

plot_decision_regions(x_train,y_train,clf=Neuron_class)
plt.show()


