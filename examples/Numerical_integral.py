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
#%%
import numpy as np

def magnetic_field_Bz(a, b, z, I):
    """
    计算矩形回路正上方 z 处的磁场 Bz
    :param a: x 方向半边长 (米)
    :param b: y 方向半边长 (米)
    :param z: 高度 (米)
    :param I: 电流强度 (安培)
    :return: Bz (单位: 特斯拉)
    """
    mu_0 = 4 * np.pi * 1e-7  # 真空磁导率 (N/A^2)

    sqrt_term = np.sqrt(a**2 + b**2 + z**2)
    denom_a = a**2 + z**2
    denom_b = b**2 + z**2

    Bz = (mu_0 * I * a * b) / (np.pi * sqrt_term) * (1/denom_a + 1/denom_b)
    return Bz

B = magnetic_field_Bz(a=2e-3, b=1.6e-3, z=0.1, I=50)
print(f"Bz = {B:.3e} T")

# %%
