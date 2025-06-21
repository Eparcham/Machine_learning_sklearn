import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.neural_network import MLPClassifier

np.random.seed(0)
# Load XOR data
x_train = []
y_train = []
with open('./data/xor.txt') as f:
    for line in f:
        s = line.strip().split(',')
        y_train.append(int(s[-1]))
        x_train.append([int(s[0]), int(s[1])])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Activation Functions
def relu(x): return np.maximum(x, 0)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Loss Functions
def bce(y, y_hat):
    eps = 1e-8
    return np.mean(-(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def accuracy(y, y_hat, t=0.5):
    y_hat_bin = (y_hat >= t).astype(int)
    return np.mean(y_hat_bin == y)

class MLP:
    def __init__(self, hidden_layers_size, output_size=1, hidden_af='relu', output_activation='sigmoid',
                 loss_fn=bce, maxit=1000, eta=0.9):
        self.hidden = hidden_layers_size
        self.output_activation = output_activation
        self.hidden_af = hidden_af
        self.loss_fn = loss_fn
        self.output_size = output_size
        self.maxit = maxit
        self.eta = eta
        self.loss_history = []

        self.ws = []
        self.bs = []

    def init_weights(self, input_size):
        layer_sizes = [input_size] + self.hidden + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.ws.append(np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in))
            self.bs.append(np.zeros(fan_out))

    def activate_fun(self, x, name):
        if name == 'relu': return relu(x)
        if name == 'sigmoid': return sigmoid(x)
        if name == 'tanh': return tanh(x)
        if name == 'softmax': return softmax(x)
        return x

    def activate_derivative(self, x, name):
        if name == 'relu': return (x > 0).astype(float)
        if name == 'sigmoid': s = sigmoid(x); return s * (1 - s)
        if name == 'tanh': return 1 - np.tanh(x) ** 2
        return 1  # for linear

    def forward(self, x):
        hs, activations = [], []
        a = x
        for i in range(len(self.ws) - 1):
            h = a @ self.ws[i] + self.bs[i]
            hs.append(h)
            a = self.activate_fun(h, self.hidden_af)
            activations.append(a)
        # Output layer
        h_out = a @ self.ws[-1] + self.bs[-1]
        hs.append(h_out)
        y_hat = self.activate_fun(h_out, self.output_activation)
        activations.append(y_hat)
        return hs, activations

    def fit(self, x, y):
        y = y if y.ndim == 2 else y[:, None]
        self.init_weights(x.shape[1])

        for epoch in range(self.maxit):
            hs, activations = self.forward(x)
            y_hat = activations[-1]
            loss = self.loss_fn(y, y_hat)
            self.loss_history.append(loss)

            # Backpropagation
            deltas = [y_hat - y]
            for i in reversed(range(len(self.hidden))):
                delta = (deltas[0] @ self.ws[i + 1].T) * self.activate_derivative(hs[i], self.hidden_af)
                deltas.insert(0, delta)

            for i in range(len(self.ws)):
                a_prev = x if i == 0 else activations[i - 1]
                self.ws[i] -= self.eta * (a_prev.T @ deltas[i]) / len(y)
                self.bs[i] -= self.eta * np.mean(deltas[i], axis=0)

            if epoch % 50 == 0:
                acc = accuracy(y, y_hat)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    def predict(self, x):
        _, activations = self.forward(x)
        return activations[-1]

    def score(self, x, y):
        y = y if y.ndim == 2 else y[:, None]
        y_hat = self.predict(x)
        return accuracy(y, y_hat)

# Train and visualize
model = MLP(hidden_layers_size=[2], output_size=1)
model.fit(x_train, y_train)

plot_decision_regions(x_train, y_train, clf=model)
plt.title("Custom MLP Decision Region")
plt.show()

print("Prediction [1,1], [0,0]:", model.predict(np.array([[1,1],[0,0]])))
print("Prediction [0,1], [1,0]:", model.predict(np.array([[0,1],[1,0]])))
