import numpy as np

# Step function (activation)
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron training function
def perceptron_train(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])  # initialize weights
    bias = 0

    for _ in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            y_pred = step_function(z)
            error = y[i] - y_pred

            # Update rule (Perceptron Learning Rule)
            weights += lr * error * X[i]
            bias += lr * error
    return weights, bias

# Example data (AND logic gate)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])  # Output for AND gate

w, b = perceptron_train(X, y)
print("Weights:", w, "Bias:", b)

# Testing
for x in X:
    print(x, "->", step_function(np.dot(x, w) + b))
