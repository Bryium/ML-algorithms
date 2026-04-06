# Linear Regression and Locally Weighted Linear Regression

import numpy as np
import matplotlib.pyplot as plt


# Generate sample data
np.random.seed(0)
X = np.linspace(-3, 3, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Add bias term
X_mat = np.c_[np.ones(len(X)), X]


# 1. Linear Regression (Best Fit Line)
theta = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y
y_pred = X_mat @ theta


# 2. Locally Weighted Linear Regression
def lwlr(test_point, X, y, k):
    m = X.shape[0]
    weights = np.eye(m)

    for i in range(m):
        diff = test_point - X[i]
        weights[i, i] = np.exp(-(diff @ diff.T) / (2 * k**2))

    theta = np.linalg.pinv(X.T @ weights @ X) @ (X.T @ weights @ y)
    return test_point @ theta

def lwlr_curve(X, y, k):
    y_lwlr = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y_lwlr[i] = lwlr(X[i], X, y, k)
    return y_lwlr

# Different k values
k_values = [0.1, 0.5, 1.0]


# Plotting
plt.scatter(X, y, label="Data")

# Linear regression line
plt.plot(X, y_pred, label="Linear Regression")

# LWLR lines
for k in k_values:
    y_lwlr = lwlr_curve(X_mat, y, k)
    plt.plot(X, y_lwlr, label=f"LWLR k={k}")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear vs Locally Weighted Regression")
plt.legend()
plt.show()