import numpy as np
import matplotlib.pyplot as plt

# Function to generate pairs of XY from input pairs of 2 arrays
def generate_xy_pairs(xy_arrays):
    x_values = np.array([pair[0] for pair in xy_arrays])
    y_values = np.array([pair[1] for pair in xy_arrays])
    return x_values, y_values

# Function to compute the coefficients for each type of regression
def compute_regression_coefficients(x_values, y_values):
    x, y = np.meshgrid(x_values, y_values)
    z = np.exp(x)  # Exponential function
    linear_coeff = np.polyfit(x, y, 1)
    quadratic_coeff = np.polyfit(x, y**2, 2)
    return linear_coeff, quadratic_coeff, z

# Function to plot the regression curves
def plot_regression_curves(x_values, y_values, z_values, coeffs):
    linear_coeff, quadratic_coeff, z = coeffs
    plt.plot(x_values, y_values, label='Data')
    plt.plot(x_values, z, label='Exponential Regression')
    plt.plot(x_values, z**2, label='Quadratic Regression')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

# Sample pairs of XY
xy_pairs = [
    (np.array([1, 2, 3]), np.array([6, 7, 8])),
    (np.array([4, 5, 6]), np.array([12, 15, 18])),
]

# Generate XY pairs
x_values, y_values = generate_xy_pairs(xy_pairs)
print("x_values shape:", x_values.shape)
print("x_values:", x_values)
print("y_values shape:", y_values.shape)
print("y_values:", y_values)

# Compute regression coefficients
linear_coeff, quadratic_coeff, z_values = compute_regression_coefficients(x_values, y_values)

# Plot the regression curves
plot_regression_curves(x_values, y_values, z_values, (linear_coeff, quadratic_coeff, z_values))