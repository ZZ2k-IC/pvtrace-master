import numpy as np
from scipy.integrate import quad

# Constants
n1 = 1.5485  # Refractive index of quartz
n2 = 1       # Refractive index of air

# Define the updated integrand based on the more compact expression
def integrand_compact(x):
    sin_x = np.sin(x)
    sin_2x = np.sin(2 * x)
    cos_x = np.cos(x)
    sqrt_term = np.sqrt(1 - (n1 * sin_x) ** 2)

    Rs = ((n1 * cos_x - sqrt_term) / (n1 * cos_x + sqrt_term)) ** 2
    Rp = ((n1 * sqrt_term - n2 * cos_x) / (n1 * sqrt_term + n2 * cos_x)) ** 2
    R = 0.5 * (Rs + Rp)
    T = 1 - R
    return T * 4/3 * sin_2x

def integrand_square(x):
    sin_x = np.sin(x)
    sin_2x = np.sin(2 * x)
    cos_x = np.cos(x)
    sqrt_term = np.sqrt(1 - (n1 * sin_x) ** 2)

    Rs = ((n1 * cos_x - sqrt_term) / (n1 * cos_x + sqrt_term)) ** 2
    Rp = ((n1 * sqrt_term - n2 * cos_x) / (n1 * sqrt_term + n2 * cos_x)) ** 2
    R = 0.5 * (Rs + Rp)
    T = 1 - R
    return (T**2) * 4/3 * sin_2x


# Integration limits
theta_c = np.arcsin(n2 / n1)  # critical angle in radians

# Perform the integration
fraction, error = quad(integrand_compact, 0, theta_c)
fraction_square, error_square = quad(integrand_square, 0, theta_c)
variance = fraction_square - fraction**2
sigma = np.sqrt(variance)

print(f"Integral result: {fraction}")
print(f"Integral result (square): {fraction_square}")
print(f"Variance: {variance}")
print(f"Standard deviation: {sigma}")
