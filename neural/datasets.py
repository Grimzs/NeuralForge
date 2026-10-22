import numpy as np


def xor_dataset():
    """Классический XOR: 4 точки, нелинейная задача."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    return X, y


def linear_dataset(n=100, noise=0.1):
    """Линейная регрессия y = 2x + 1."""
    X = np.random.rand(n, 1) * 10 - 5
    y = 2 * X + 1 + np.random.randn(n, 1) * noise
    return X, y


def circle_dataset(n=200, noise=0.1):
    """Точка внутри/снаружи круга — бинарная классификация."""
    X = np.random.randn(n, 2) * 2
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) > 2.5).astype(float).reshape(-1, 1)
    return X, y


def spiral_dataset(n_points=200, noise=0.1):
    """Две закрученные спирали — сложная нелинейная задача."""
    n = n_points // 2
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0.1, 1, n)
    x1 = np.c_[r * np.cos(theta), r * np.sin(theta)]
    x2 = np.c_[-r * np.cos(theta), -r * np.sin(theta)]
    x1 += np.random.randn(*x1.shape) * noise
    x2 += np.random.randn(*x2.shape) * noise
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n), np.ones(n)]).reshape(-1, 1)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]