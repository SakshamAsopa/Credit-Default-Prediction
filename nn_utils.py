import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def forward_propagation(X, params):
    W1, b1, W2, b2 = params
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return A2

def predict_proba(X, params):
    return forward_propagation(X, params)

def predict(X, params, threshold=0.5):
    probs = predict_proba(X, params)
    return (probs > threshold).astype(int)
