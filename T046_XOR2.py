from random import random
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
from sklearn.neural_network import MLPClassifier,MLPRegressor

x_train = []
y_train = []
with open('./data/xor.txt') as f:
    lines = f.readlines()
    for l in lines:
        if l[-1] == '\n':
            s = l[:-1].split(',')
        else:
            s = l.split(',')
        y_train.append(int(s[-1]))
        x_train.append([int(s[0]), int(s[1])])
        a = 0

x_train = np.array(x_train)
y_train = np.array(y_train)[:,None]

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

def mlp_xor(x,w1,b1,w2,b2):
    hidden_output = np.dot(x,w1)+b1
    activation = relu(hidden_output)
    output = sigmoid(np.dot(activation,w2)+b2)
    return output,hidden_output,activation

np.random.seed(40)
w1 = np.random.rand(2,2)
b1 = np.random.rand(1,2)
w2 = np.random.rand(2,1)
b2 = np.random.rand(1)

learning_rate = 0.1
num_epochs = 1000

for i in range(num_epochs):
    y_hat,h,a = mlp_xor(x_train,w1,b1,w2,b2)

    loss = bce(y_train,y_hat)
    print(loss)

    delta_output = y_hat - y_train
    w2_grad = (a.T @ delta_output)/len(y_train)
    b2_grad = delta_output.mean(0)
    w2 -= learning_rate * w2_grad
    b2 -= learning_rate * b2_grad

    delta_hidden = (delta_output @ w2.T) * (h>0)
    w1_grad = (x_train.T @ delta_hidden)/len(y_train)
    b1_grad = delta_hidden.mean(0)
    w1 -= learning_rate * w1_grad
    b1 -= learning_rate * b1_grad


