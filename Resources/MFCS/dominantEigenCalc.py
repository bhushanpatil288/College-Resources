import numpy as np

# Define the matrix A
A = np.array([[5, 4, 2], [1, 7, 6], [3, 1, 8]])

# Initial vector
x0 = np.array([1, 0, 0])

# Power iteration process
def power_iteration(A, x0, num_iterations=3):
    x = x0
    eigenvalue_estimates = []

    for _ in range(num_iterations):
        # Multiply by the matrix A
        x = np.dot(A, x)

        # Normalize the vector
        x = x / np.linalg.norm(x)

        # Estimate the eigenvalue using Rayleigh quotient
        eigenvalue = np.dot(x.T, np.dot(A, x)) / np.dot(x.T, x)
        eigenvalue_estimates.append(eigenvalue)

    return x, eigenvalue_estimates[-1]

# Run the power iteration for 3 iterations
eigenvector, dominant_eigenvalue = power_iteration(A, x0, num_iterations=3)

print("Dominant Eigenvalue:", dominant_eigenvalue)
print("Dominant Eigenvector:", eigenvector)
