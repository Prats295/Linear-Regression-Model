import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("C:/Users/bncoe/MY PROGRAM/ML/data.txt", delimiter=',')
X = data[:, 0]
Y = data[:, 1]

# Reshape Y to column vector
Y = Y.reshape(-1, 1)

# Add intercept term (bias) to X
X = np.vstack((np.ones(X.shape[0]), X)).T  # shape (m, 2)

# Plot the data
plt.scatter(X[:, 1], Y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Data Scatter Plot")
plt.show()

# Gradient Descent Function
def model(X, Y, learning_rate, iteration):
    m = Y.size
    theta = np.zeros((2, 1))
    cost_list = []
    for i in range(iteration):
        y_pred = np.dot(X, theta)
        cost = (1 / (2 * m)) * np.sum((y_pred - Y) ** 2)
        d_theta = (1 / m) * np.dot(X.T, (y_pred - Y))
        theta = theta - learning_rate * d_theta
        cost_list.append(cost)
    return theta, cost_list

# Parameters
iteration = 100
learning_rate = 0.00000005

# Train model
theta, cost_list = model(X, Y, learning_rate=learning_rate, iteration=iteration)

# Print final parameters
print("Final theta values:\n", theta)

# Plot cost over iterations
plt.plot(range(iteration), cost_list)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over Iterations")
plt.show()
# Plot the original data
plt.scatter(X[:, 1], Y, color='blue', label='Data')

# Predicted values using the learned theta
y_pred_line = np.dot(X, theta)

# Plot regression line
plt.plot(X[:, 1], y_pred_line, color='red', label='Regression Line')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
# Unpack theta
theta0 = theta[0, 0]
theta1 = theta[1, 0]

# Print in human-readable format
print("Intercept (θ₀/c):", round(theta0, 4))
print("Slope     (θ₁/m):", round(theta1, 4))
x_new = 2500
y_pred = theta0 + theta1 * x_new
print(f"Predicted Y for X={x_new}:", round(y_pred, 2))
