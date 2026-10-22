"""
Многослойный перцептрон с самообучением (backprop).
Только NumPy, без внешних зависимостей.
"""
import json
import numpy as np


# ============================================================
# Activation functions
# ============================================================
class Activation:
    @staticmethod
    def sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    @staticmethod
    def relu(z):     return np.maximum(0, z)
    @staticmethod
    def tanh(z):     return np.tanh(z)
    @staticmethod
    def linear(z):   return z
    @staticmethod
    def softmax(z):
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid_deriv(z, a): return a * (1 - a)
    @staticmethod
    def relu_deriv(z, a):    return (z > 0).astype(float)
    @staticmethod
    def tanh_deriv(z, a):    return 1 - a ** 2
    @staticmethod
    def linear_deriv(z, a):  return np.ones_like(z)

    @classmethod
    def forward(cls, name, z):
        return getattr(cls, name)(z)

    @classmethod
    def derivative(cls, name, z, a):
        return getattr(cls, name + "_deriv")(z, a)


# ============================================================
# Layer
# ============================================================
class Layer:
    """Слой нейронов: W (n_in × n_out), b (n_out), activation."""

    def __init__(self, n_inputs, n_neurons, activation="relu"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        # Xavier/Glorot init
        limit = np.sqrt(6.0 / (n_inputs + n_neurons))
        self.W = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        self.b = np.zeros(n_neurons)

        self._x = self._z = self._a = None  # кэш для backprop

    def forward(self, X):
        self._x = X
        self._z = X @ self.W + self.b
        self._a = Activation.forward(self.activation, self._z)
        return self._a

    def backward(self, grad_a, lr):
        """grad_a — dL/da. Возвращает dL/dx для предыдущего слоя."""
        if self.activation == "softmax":
            delta = grad_a  # комбинированный softmax+CE уже снаружи
        else:
            delta = grad_a * Activation.derivative(self.activation, self._z, self._a)

        dW = self._x.T @ delta
        db = delta.sum(axis=0)
        grad_x = delta @ self.W.T

        self.W -= lr * dW
        self.b -= lr * db
        return grad_x

    def __repr__(self):
        return f"Layer({self.n_inputs}→{self.n_neurons}, {self.activation})"


# ============================================================
# NeuralNetwork
# ============================================================
class NeuralNetwork:
    """Многослойный перцептрон с самообучением."""

    def __init__(self, layer_sizes, hidden_activation="relu",
                 output_activation="sigmoid", loss="mse", seed=42):
        """
        layer_sizes: [n_inputs, h1, h2, ..., n_outputs]
        loss: 'mse' | 'bce'
        """
        np.random.seed(seed)
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_name = loss

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_output = (i == len(layer_sizes) - 2)
            act = output_activation if is_output else hidden_activation
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], act))

        self.loss_history = []

    # ── Forward ──
    def forward(self, X):
        a = np.asarray(X, dtype=float)
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # ── Loss + gradient по выходу ──
    def _compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        if self.loss_name == "mse":
            loss = np.mean((y_pred - y_true) ** 2)
            grad = 2.0 * (y_pred - y_true) / m
        elif self.loss_name == "bce":
            eps = 1e-9
            yp = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y_true * np.log(yp) + (1 - y_true) * np.log(1 - yp))
            # sigmoid + BCE → объединённый градиент
            grad = (y_pred - y_true) / m
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")
        return float(loss), grad

    # ── Один шаг обучения ──
    def _train_step(self, X, y, lr):
        y_pred = self.forward(X)
        loss, grad = self._compute_loss(y_pred, y)

        for i, layer in enumerate(reversed(self.layers)):
            is_last = (i == 0)
            # bce + sigmoid → derivative уже учтён в grad
            if is_last and self.loss_name == "bce" and layer.activation == "sigmoid":
                delta = grad
                dW = layer._x.T @ delta
                db = delta.sum(axis=0)
                grad = delta @ layer.W.T
                layer.W -= lr * dW
                layer.b -= lr * db
            else:
                grad = layer.backward(grad, lr)
        return loss

    # ── Обучение ──
    def fit(self, X, y, epochs=1000, lr=0.1, batch_size=None,
            verbose=False, log_every=None, early_stop=None):
        """
        batch_size=None — full batch
        batch_size=1    — онлайн-обучение (SGD)
        batch_size=N    — мини-батчи
        early_stop=N    — остановка, если loss не падает N эпох
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.loss_history = []
        best, stale = float("inf"), 0

        for epoch in range(epochs):
            if batch_size:
                idx = np.random.permutation(len(X))
                losses = []
                for i in range(0, len(X), batch_size):
                    batch = idx[i:i + batch_size]
                    losses.append(self._train_step(X[batch], y[batch], lr))
                loss = float(np.mean(losses))
            else:
                loss = self._train_step(X, y, lr)

            self.loss_history.append(loss)

            if verbose and (
                (log_every and epoch % log_every == 0) or
                (not log_every and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1))
            ):
                print(f"Эпоха {epoch+1:4d}/{epochs} — loss: {loss:.6f}")

            if early_stop:
                if loss < best - 1e-6:
                    best, stale = loss, 0
                else:
                    stale += 1
                    if stale >= early_stop:
                        if verbose:
                            print(f"Ранняя остановка на эпохе {epoch+1}")
                        break
        return self

    # ── Predict ──
    def predict(self, X):
        return self.forward(X)

    def predict_class(self, X, threshold=0.5):
        return (self.predict(X) > threshold).astype(int)

    # ── Save/Load ──
    def save(self, path):
        data = {
            "layer_sizes": self.layer_sizes,
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "loss": self.loss_name,
            "layers": [
                {"W": l.W.tolist(), "b": l.b.tolist(), "activation": l.activation}
                for l in self.layers
            ],
            "loss_history": self.loss_history,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        net = cls(data["layer_sizes"], data["hidden_activation"],
                  data["output_activation"], data["loss"])
        for layer, ld in zip(net.layers, data["layers"]):
            layer.W = np.array(ld["W"])
            layer.b = np.array(ld["b"])
            layer.activation = ld["activation"]
        net.loss_history = data["loss_history"]
        return net

    # ── Info ──
    def summary(self):
        lines = [f"NeuralNetwork({self.layer_sizes}) — loss={self.loss_name}"]
        total = 0
        for i, l in enumerate(self.layers):
            p = l.W.size + l.b.size
            total += p
            lines.append(f"  [{i}] {l.n_inputs:3d} → {l.n_neurons:3d}  {l.activation:8s}  params={p}")
        lines.append(f"  Всего параметров: {total}")
        return "\n".join(lines)