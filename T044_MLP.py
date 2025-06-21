import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

# ────── Activation Functions ────── #
def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ────── Loss Functions ────── #
def bce(y, y_hat):
    eps = 1e-8
    return np.mean(-(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)))

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def accuracy(y, y_hat, threshold=0.5):
    y_pred = (y_hat >= threshold).astype(int)
    return np.mean(y_pred == y)

# ────── MLP Class ────── #
class MLP:
    def __init__(self, hidden_layers_size, output_size=1, hidden_af='relu',
                 output_activation='sigmoid', loss_fn=bce, maxit=2000, eta=0.01):
        self.hidden_layers = hidden_layers_size
        self.output_size = output_size
        self.hidden_af = hidden_af
        self.output_af = output_activation
        self.loss_fn = loss_fn
        self.maxit = maxit
        self.eta = eta
        self.loss_history = []

        self.ws = []
        self.bs = []

    def _activation(self, x, kind):
        if kind == 'relu': return relu(x)
        if kind == 'sigmoid': return sigmoid(x)
        if kind == 'tanh': return tanh(x)
        if kind == 'softmax': return softmax(x)
        return x

    def _activation_derivative(self, x, kind):
        if kind == 'relu': return (x > 0).astype(float)
        if kind == 'sigmoid':
            s = sigmoid(x)
            return s * (1 - s)
        if kind == 'tanh': return 1 - np.tanh(x) ** 2
        return np.ones_like(x)

    def init_weights(self, input_size):
        layer_sizes = [input_size] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.ws.append(np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in))
            self.bs.append(np.zeros(fan_out))

    def forward(self, x):
        hs, activations = [], []
        a = x
        for i in range(len(self.ws) - 1):
            h = a @ self.ws[i] + self.bs[i]
            hs.append(h)
            a = self._activation(h, self.hidden_af)
            activations.append(a)
        # Output layer
        h_out = a @ self.ws[-1] + self.bs[-1]
        hs.append(h_out)
        y_hat = self._activation(h_out, self.output_af)
        activations.append(y_hat)
        return hs, activations

    def backward(self, x, y, hs, activations):
        y = y if y.ndim == 2 else y[:, None]
        delta = activations[-1] - y
        grads_w = []
        grads_b = []

        for i in reversed(range(len(self.ws))):
            a_prev = x if i == 0 else activations[i - 1]
            dw = a_prev.T @ delta / len(y)
            db = np.mean(delta, axis=0)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.ws[i].T) * self._activation_derivative(hs[i - 1], self.hidden_af)

        # Update weights and biases
        for i in range(len(self.ws)):
            self.ws[i] -= self.eta * grads_w[i]
            self.bs[i] -= self.eta * grads_b[i]

    def fit(self, x, y):
        y = y[:, None] if y.ndim == 1 else y
        self.init_weights(x.shape[1])
        for epoch in range(self.maxit):
            hs, activations = self.forward(x)
            loss = self.loss_fn(y, activations[-1])
            acc = accuracy(y, activations[-1])
            self.loss_history.append(loss)
            self.backward(x, y, hs, activations)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")

    def predict(self, x):
        _, activations = self.forward(x)
        return activations[-1]

    def score(self, x, y):
        y = y[:, None] if y.ndim == 1 else y
        return accuracy(y, self.predict(x))

    def parameters(self):
        return {'weights': self.ws, 'biases': self.bs}

# ────── Generate Data ────── #
x, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1.0, random_state=5)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ────── Train Model ────── #
model = MLP(hidden_layers_size=[4, 3, 3], output_size=1, loss_fn=bce)
model.fit(x_train, y_train)
print(f"Test Accuracy: {model.score(x_test, y_test):.4f}")

# ────── Plot Decision Regions ────── #
plot_decision_regions(x_train, y_train, clf=model)
plt.title("Decision Boundary - Custom MLP")
plt.show()
