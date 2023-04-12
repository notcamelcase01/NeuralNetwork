import numpy as np


def g(z):
    return 1/(1 + np.exp(-z))


def dg(z):
    return g(z) * (1 - g(z))


def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return np.heaviside(z, 1)


def L(a, y):
    dep = 1e-10
    x = np.maximum(1e-10, a)
    y0 = np.maximum(1e-10, 1 - a)
    return -(y * np.log(x) + (1 - y) * np.log(y0))


def da(a, y):
    return (a - y)/(a*(1 - a))


def cost(a, y, m):
    return np.sum(L(a, y))/m


def forward_propagation(x, w1, w2, b1, b2):
    z1 = w1 @ x + b1
    a1 = relu(z1)
    z2 = w2 @ a1 + b2
    a2 = g(z2)
    return z1, z2, a1, a2


def backward_propagation(x, y, z1, z2, a1, a2, w2, m):
    dz2 = da(a2, y) * dg(z2)
    dw2 = 1/m * dz2 @ a1.T
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    da1 = w2.T @ dz2
    dz1 = da1 * drelu(z1)
    dw1 = 1/m * dz1 @ x.T
    db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, dw2, db1, db2


def descent(x, y, max_iter, alpha, n0, n1, n2, m):
    w1 = np.random.randn(n1, n0) * 0.01
    b1 = np.random.randn(n1, 1)
    w2 = np.random.randn(n2, n1) * 0.01
    b2 = np.random.randn(n2, 1)
    for i in range(max_iter):
        z1, z2, a1, a2 = forward_propagation(x, w1, w2, b1, b2)
        dw1, dw2, db1, db2 = backward_propagation(x, y, z1, z2, a1, a2, w2, m)
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        b1 = b1 - alpha * db1
        b2 = b2 - alpha * db2
        if i % 500 == 0:
            h = cost(a2, y, m)
            print(h)
            if h < 10**(-2):
                print(h)
                break
    return w1, w2, b1, b2
