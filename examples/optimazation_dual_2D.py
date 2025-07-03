import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# Parameters
n_wg = 1.45         # waveguide refractive index
n_abs = 1.64               # absorber refractive index
alpha = 6.4             # absorption coefficient
rectangle_width = 15 
rectangle_height = 0.5
layer_gap = 0.0          # Gap between top and bottom layers
num_rays = 500            # number of rays

# Total system height includes gap
total_height = 2 * rectangle_height + layer_gap

def create_interface_curve():
    # Create 20 control points for the interface
    x_control = np.linspace(0, rectangle_width, 20)
    # Initialize with diagonal line, can be modified for optimization
    y_control = rectangle_height*1 - x_control * rectangle_height / rectangle_width
    # Ensure boundary conditions
    y_control[0] = rectangle_height*1
    y_control[-1] = 0
    
    # Create cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')
    return spline, x_control, y_control
"""
def create_interface_curve():
    # Create 20 control points for the interface
    x_control = np.linspace(0, rectangle_width, 20)
    
    # SIGMOID CURVE PARAMETERS
    # Sigmoid: y = A / (1 + exp(-k*(x - x0))) + B
    A = rectangle_height          # Amplitude (height difference)
    k = 0.1                      # Steepness factor (higher = steeper transition)
    x0 = rectangle_width * 0.5   # Inflection point (where curve is steepest)
    B = 0                        # Vertical offset
    
    # Generate sigmoid curve - FLIPPED to start high and end low
    y_control = A / (1 + np.exp(k * (x_control - x0))) + B
    
    # Ensure boundary conditions
    y_control[0] = rectangle_height   # Start at top
    y_control[-1] = 0                 # End at bottom
    
    # Create cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')
    return spline, x_control, y_control
"""
"""
def create_interface_curve():
    # Create 20 control points for the interface
    x_control = np.linspace(0, rectangle_width, 20)
    
    # Create exponential decay curve for first 19 points
    # Exponential decay: y = A * exp(-k*x) + B
    # where A controls amplitude, k controls decay rate, B is offset
    
    # Parameters for exponential decay
    A = rectangle_height * 0.9  # Amplitude (adjust for steepness)
    k = 5.0  # Decay rate (higher = faster decay)
    B = rectangle_height * 0.0  # Vertical offset
    
    # Generate exponential decay for first 19 points
    x_exp = x_control[:-1]  # All points except the last one
    y_control = A * np.exp(-k * x_exp / rectangle_width) + B
    
    # Ensure boundary conditions
    y_control[0] = rectangle_height  # Start at top
    
    # Add the final point manually
    y_control = np.append(y_control, 0)  # End at bottom
    
    # Create cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')
    return spline, x_control, y_control
"""

# Fresnel and utility functions (same as before)
def fresnel_coefficients(theta_i, n1, n2):
    """Calculate Fresnel coefficients for s-polarized light"""
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    
    sin_t_squared = (n1/n2)**2 * sin_i**2
    if sin_t_squared > 1:
        return 1.0, 0.0, None  # Total internal reflection
    
    cos_t = np.sqrt(1 - sin_t_squared)
    
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    t_s = 2*n1*cos_i / (n1*cos_i + n2*cos_t)
    
    R = r_s**2
    T = (n2*cos_t)/(n1*cos_i) * t_s**2
    
    theta_t = np.arcsin((n1/n2) * sin_i)
    
    return R, T, theta_t

def lambertian_sample_theta(n, theta_max=np.pi*43/180):
    p = np.random.uniform(0, 1, n)
    theta = np.arcsin(np.sqrt(p) * np.sin(theta_max))
    return theta

def generate_rays_lambertian(n):
    """Generate rays with Lambertian distribution pointing BOTH upward and downward"""
    thetas = lambertian_sample_theta(n)
    rays = []
    for theta in thetas:
        # Start rays in top layer
        y_start = rectangle_height + np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        
        # RANDOMLY choose upward or downward direction
        if np.random.random() < 0.5:
            # Upward pointing rays (original)
            direction = np.array([np.cos(theta), np.sin(theta)])
        else:
            # Downward pointing rays (flipped)
            direction = np.array([np.cos(theta), -np.sin(theta)])
        
        intensity = 1.0
        rays.append([origin, direction, intensity])
    return rays

def generate_rays_lambertian_dual(n):
    """Generate rays for BOTH top and bottom layers with bidirectional Lambertian distribution"""
    rays = []
    rays_per_layer = n // 2  # Split rays between layers
    
    # Generate rays for TOP layer
    thetas_top = lambertian_sample_theta(rays_per_layer)
    for theta in thetas_top:
        # Top layer: y > rectangle_height + layer_gap
        y_start = rectangle_height + np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        
        # RANDOMLY choose upward or downward direction
        if np.random.random() < 0.5:
            # Upward pointing rays (original)
            direction = np.array([np.cos(theta), np.sin(theta)])
        else:
            # Downward pointing rays (flipped)
            direction = np.array([np.cos(theta), -np.sin(theta)])
        
        intensity = 1.0
        rays.append([origin, direction, intensity])
    
    # Generate rays for BOTTOM layer
    thetas_bottom = lambertian_sample_theta(rays_per_layer)
    for theta in thetas_bottom:
        # Bottom layer: 0 < y < rectangle_height
        y_start = np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        
        # RANDOMLY choose upward or downward direction
        if np.random.random() < 0.5:
            # Upward pointing rays
            direction = np.array([np.cos(theta), np.sin(theta)])
        else:
            # Downward pointing rays
            direction = np.array([np.cos(theta), -np.sin(theta)])
        
        intensity = 1.0
        rays.append([origin, direction, intensity])
    
    return rays

def generate_rays_collimated(n, angle_deg=0):
    """
    Generate parallel rays with a specific angle
    angle_deg: 0 = horizontal (rightward), 90 = vertical (upward)
    """
    rays = []
    angle_rad = np.radians(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    for i in range(n):
        # Distribute ray origins uniformly along the left boundary
        y_start = rectangle_height + layer_gap + np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        intensity = 1.0
        rays.append([origin, direction, intensity])
    
    return rays

def generate_rays_collimated_dual(n, angle_deg=0):
    """Generate parallel rays for both layers"""
    rays = []
    rays_per_layer = n // 2
    angle_rad = np.radians(angle_deg)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    # Top layer rays
    for i in range(rays_per_layer):
        y_start = rectangle_height + layer_gap + i * rectangle_height / (rays_per_layer - 1)
        origin = np.array([0.0, y_start])
        rays.append([origin, direction, 1.0])
    
    # Bottom layer rays
    for i in range(rays_per_layer):
        y_start = i * rectangle_height / (rays_per_layer - 1)
        origin = np.array([0.0, y_start])
        rays.append([origin, direction, 1.0])
    
    return rays

angle_degree_range = np.array([ 0.125,  0.375,  0.625,  0.875,  1.125,  1.375,  1.625,  1.875,
        2.125,  2.375,  2.625,  2.875,  3.125,  3.375,  3.625,  3.875,
        4.125,  4.375,  4.625,  4.875,  5.125,  5.375,  5.625,  5.875,
        6.125,  6.375,  6.625,  6.875,  7.125,  7.375,  7.625,  7.875,
        8.125,  8.375,  8.625,  8.875,  9.125,  9.375,  9.625,  9.875,
       10.125, 10.375, 10.625, 10.875, 11.125, 11.375, 11.625, 11.875,
       12.125, 12.375, 12.625, 12.875, 13.125, 13.375, 13.625, 13.875,
       14.125, 14.375, 14.625, 14.875, 15.125, 15.375, 15.625, 15.875,
       16.125, 16.375, 16.625, 16.875, 17.125, 17.375, 17.625, 17.875,
       18.125, 18.375, 18.625, 18.875, 19.125, 19.375, 19.625, 19.875,
       20.125, 20.375, 20.625, 20.875, 21.125, 21.375, 21.625, 21.875,
       22.125, 22.375, 22.625, 22.875, 23.125, 23.375, 23.625, 23.875,
       24.125, 24.375, 24.625, 24.875, 25.125, 25.375, 25.625, 25.875,
       26.125, 26.375, 26.625, 26.875, 27.125, 27.375, 27.625, 27.875,
       28.125, 28.375, 28.625, 28.875, 29.125, 29.375, 29.625, 29.875,
       30.125, 30.375, 30.625, 30.875, 31.125, 31.375, 31.625, 31.875,
       32.125, 32.375, 32.625, 32.875, 33.125, 33.375, 33.625, 33.875,
       34.125, 34.375, 34.625, 34.875, 35.125, 35.375, 35.625, 35.875,
       36.125, 36.375, 36.625, 36.875, 37.125, 37.375, 37.625, 37.875,
       38.125, 38.375, 38.625, 38.875, 39.125, 39.375, 39.625, 39.875,
       40.125, 40.375, 40.625, 40.875, 41.125, 41.375, 41.625, 41.875,
       42.125, 42.375, 42.625, 42.875, 43.125, 43.375, 43.625, 43.875,
       44.125, 44.375, 44.625, 44.875, 45.125, 45.375, 45.625, 45.875,
       46.125, 46.375, 46.625, 46.875, 47.125, 47.375, 47.625, 47.875,
       48.125, 48.375, 48.625, 48.875, 49.125, 49.375, 49.625, 49.875,
       50.125, 50.375, 50.625, 50.875, 51.125, 51.375, 51.625, 51.875,
       52.125, 52.375, 52.625, 52.875, 53.125, 53.375, 53.625, 53.875,
       54.125, 54.375, 54.625, 54.875, 55.125, 55.375, 55.625, 55.875,
       56.125, 56.375, 56.625, 56.875, 57.125, 57.375, 57.625, 57.875,
       58.125, 58.375, 58.625, 58.875, 59.125, 59.375, 59.625, 59.875,
       60.125, 60.375, 60.625, 60.875, 61.125, 61.375, 61.625, 61.875,
       62.125, 62.375, 62.625, 62.875, 63.125, 63.375, 63.625, 63.875,
       64.125, 64.375, 64.625, 64.875, 65.125, 65.375, 65.625, 65.875,
       66.125, 66.375, 66.625, 66.875, 67.125, 67.375, 67.625, 67.875,
       68.125, 68.375, 68.625, 68.875, 69.125, 69.375, 69.625, 69.875,
       70.125, 70.375, 70.625, 70.875, 71.125, 71.375, 71.625, 71.875,
       72.125, 72.375, 72.625, 72.875, 73.125, 73.375, 73.625, 73.875,
       74.125, 74.375, 74.625, 74.875, 75.125, 75.375, 75.625, 75.875,
       76.125, 76.375, 76.625, 76.875, 77.125, 77.375, 77.625, 77.875,
       78.125, 78.375, 78.625, 78.875, 79.125, 79.375, 79.625, 79.875,
       80.125, 80.375, 80.625, 80.875, 81.125, 81.375, 81.625, 81.875,
       82.125, 82.375, 82.625, 82.875, 83.125, 83.375, 83.625, 83.875,
       84.125, 84.375, 84.625, 84.875, 85.125, 85.375, 85.625, 85.875,
       86.125, 86.375, 86.625, 86.875, 87.125, 87.375, 87.625, 87.875,
       88.125, 88.375, 88.625, 88.875, 89.125, 89.375, 89.625, 89.875])

angle_degree_possibility = np.array([1.25065659e-04, 4.25223242e-04, 9.00472748e-04, 1.20063033e-03,
       1.52580105e-03, 1.32569599e-03, 2.10110308e-03, 2.62637885e-03,
       2.75144451e-03, 2.82648390e-03, 3.35175967e-03, 4.10215363e-03,
       4.12716676e-03, 4.67745566e-03, 5.40283649e-03, 4.77750819e-03,
       5.52790215e-03, 6.20325671e-03, 5.05265264e-03, 6.87861127e-03,
       6.55344056e-03, 6.95365067e-03, 7.07871633e-03, 6.75354561e-03,
       8.47945171e-03, 7.85412341e-03, 8.20430726e-03, 9.20483254e-03,
       8.45443858e-03, 9.27987193e-03, 9.70509517e-03, 9.65506891e-03,
       1.03554366e-02, 9.70509517e-03, 1.04554891e-02, 1.03304235e-02,
       1.07056205e-02, 1.11058306e-02, 1.09307386e-02, 1.15560669e-02,
       1.19062508e-02, 1.29317892e-02, 1.22314215e-02, 1.26316316e-02,
       1.30568548e-02, 1.33570124e-02, 1.29317892e-02, 1.38572751e-02,
       1.35070912e-02, 1.44575902e-02, 1.47327347e-02, 1.46576953e-02,
       1.42574852e-02, 1.45326296e-02, 1.64836539e-02, 1.68588509e-02,
       1.64586408e-02, 1.68088246e-02, 1.69589034e-02, 1.53330499e-02,
       1.63836014e-02, 1.59583781e-02, 1.55831812e-02, 1.52329973e-02,
       1.51329448e-02, 1.50328923e-02, 1.46576953e-02, 1.38822882e-02,
       1.37071963e-02, 1.39573276e-02, 1.34820781e-02, 1.28067235e-02,
       1.29568023e-02, 1.28317367e-02, 1.20063033e-02, 1.16311063e-02,
       1.35821306e-02, 1.13559619e-02, 1.29818155e-02, 1.16311063e-02,
       1.22064084e-02, 1.26566447e-02, 1.10307912e-02, 1.02303709e-02,
       1.19062508e-02, 1.01053053e-02, 1.13809750e-02, 9.80514770e-03,
       9.98023963e-03, 9.17981941e-03, 9.32989820e-03, 9.83016083e-03,
       8.77960929e-03, 7.52895270e-03, 5.97813852e-03, 5.52790215e-03,
       5.05265264e-03, 4.52737687e-03, 3.62690412e-03, 3.12664149e-03,
       2.80147077e-03, 2.32622127e-03, 2.07608995e-03, 1.70089297e-03,
       1.35070912e-03, 9.25485880e-04, 9.25485880e-04, 7.25380825e-04,
       5.50288902e-04, 5.75302034e-04, 3.75196978e-04, 3.75196978e-04,
       3.00157583e-04, 1.00052528e-04, 2.25118187e-04, 2.00105055e-04,
       1.25065659e-04, 1.25065659e-04, 1.25065659e-04, 1.25065659e-04,
       7.50393957e-05, 5.00262638e-05, 2.50131319e-05, 5.00262638e-05,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])

def generate_rays_customized(n):
    """Generate rays using custom angle probability distribution with bidirectional rays"""
    rays = []
    
    # Convert angles to radians
    angles_rad = np.radians(angle_degree_range)
    
    # Normalize probabilities to ensure they sum to 1
    probabilities = angle_degree_possibility / np.sum(angle_degree_possibility)
    
    # Generate n rays
    for i in range(n):
        # Sample angle based on probability distribution
        angle_idx = np.random.choice(len(angles_rad), p=probabilities)
        theta = angles_rad[angle_idx]
        
        # Start rays in top layer (you can modify this)
        y_start = rectangle_height + np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        
        # RANDOMLY choose upward or downward direction (50/50 split)
        if np.random.random() < 0.5:
            # Upward pointing rays (positive y-component)
            direction = np.array([np.cos(theta), np.sin(theta)])
        else:
            # Downward pointing rays (negative y-component)
            direction = np.array([np.cos(theta), -np.sin(theta)])
        
        intensity = 1.0
        rays.append([origin, direction, intensity])
    
    return rays


def get_surface_normal(spline, x):
    dy_dx = spline.derivative()(x)
    normal = np.array([-dy_dx, 1])
    normal = normal / np.linalg.norm(normal)
    return normal

# Replace the determine_region function:
def determine_region(point, spline_top, spline_bottom):
    """Determine which region a point is in - NO AIR GAP"""
    x, y = point
    
    if y > rectangle_height:
        # Top layer (starts at y = rectangle_height)
        if y > spline_top(x) + rectangle_height:
            return 'top_absorber'
        else:
            return 'top_waveguide'
    else:
        # Bottom layer (0 <= y <= rectangle_height)
        if y > spline_bottom(x):
            return 'bottom_absorber'
        else:
            return 'bottom_waveguide'

def intersect_interface(ray, spline, layer_offset):
    """Find intersection with interface curve (offset for top/bottom layer)"""
    origin, direction, _ = ray
    
    def intersection_equation(t):
        if t <= 1e-6:
            return float('inf')
        point = origin + t * direction
        x, y = point
        if x < 0 or x > rectangle_width:
            return float('inf')
        try:
            spline_y = spline(x) + layer_offset
            return y - spline_y
        except:
            return float('inf')
    
    t_max = rectangle_width
    dt = 0.05
    t_current = dt
    prev_val = intersection_equation(t_current)
    
    while t_current < t_max:
        t_current += dt
        current_val = intersection_equation(t_current)
        
        if (not np.isinf(prev_val) and not np.isinf(current_val) and 
            prev_val * current_val < 0):
            try:
                t_intersect = fsolve(intersection_equation, t_current - dt/2)[0]
                if t_intersect > 1e-6:
                    intersection_point = origin + t_intersect * direction
                    x_int, y_int = intersection_point
                    if 0 <= x_int <= rectangle_width:
                        return intersection_point, t_intersect
            except:
                pass
        
        prev_val = current_val
    
    return None, None

# Replace the intersect_layer_boundaries function:
def intersect_layer_boundaries(ray):
    """Find intersections with layer boundaries - NO GAP"""
    origin, direction, _ = ray
    intersections = []
    
    # Layer boundaries (no gap)
    boundaries = [
        (0, 'bottom_boundary'),                    # y = 0 (bottom of system)
        (rectangle_height, 'layer_interface'),     # y = 0.5 (interface between layers)
        (total_height, 'top_boundary')             # y = 1.0 (top of system)
    ]
    
    for y_boundary, boundary_type in boundaries:
        if abs(direction[1]) > 1e-10:  # Ray not horizontal
            t = (y_boundary - origin[1]) / direction[1]
            if t > 1e-6:
                x_int = origin[0] + t * direction[0]
                if 0 <= x_int <= rectangle_width:
                    intersections.append((np.array([x_int, y_boundary]), t, boundary_type))
    
    # Side boundaries (x = 0 and x = rectangle_width)
    if direction[0] > 0:  # Moving right
        t = (rectangle_width - origin[0]) / direction[0]
        if t > 1e-6:
            y_int = origin[1] + t * direction[1]
            if 0 <= y_int <= total_height:
                intersections.append((np.array([rectangle_width, y_int]), t, 'right_boundary'))
    
    if direction[0] < 0:  # Moving left
        t = -origin[0] / direction[0]
        if t > 1e-6:
            y_int = origin[1] + t * direction[1]
            if 0 <= y_int <= total_height:
                intersections.append((np.array([0, y_int]), t, 'left_boundary'))
    
    return intersections

# Add this new function to track ray paths:
def simulate_dual_layer_ray_with_paths(ray, spline_top, spline_bottom, max_bounces=100):
    """Simulate ray in dual-layer system and return both absorption and ray paths"""
    absorbed_points = []
    ray_paths = []  # Store all ray segments
    current_ray = ray.copy()
    
    for bounce in range(max_bounces):
        origin, direction, intensity = current_ray
        
        # Determine current region
        current_region = determine_region(origin, spline_top, spline_bottom)
        
        # Find all possible intersections
        intersections = []
        
        # Interface intersections
        if current_region in ['top_absorber', 'top_waveguide']:
            int_point, t_int = intersect_interface(current_ray, spline_top, rectangle_height + layer_gap)
            if int_point is not None:
                intersections.append((int_point, t_int, 'top_interface'))
        
        if current_region in ['bottom_absorber', 'bottom_waveguide']:
            int_point, t_int = intersect_interface(current_ray, spline_bottom, 0)
            if int_point is not None:
                intersections.append((int_point, t_int, 'bottom_interface'))
        
        # Layer boundary intersections
        boundary_intersections = intersect_layer_boundaries(current_ray)
        intersections.extend(boundary_intersections)
        
        if not intersections:
            break
        
        # Sort by distance and take closest
        intersections.sort(key=lambda x: x[1])
        intersection_point, t, intersection_type = intersections[0]
        
        # Store ray path segment
        ray_paths.append({
            'start': origin.copy(),
            'end': intersection_point.copy(),
            'region': current_region,
            'intensity': intensity,
            'bounce': bounce
        })
        
        # Handle absorption along path (same as before)
        if current_region in ['top_absorber', 'bottom_absorber']:
            num_samples = max(10, int(t * 100))
            t_samples = np.linspace(0, t, num_samples)
            
            for i, t_sample in enumerate(t_samples[1:]):
                sample_point = origin + t_sample * direction
                sample_region = determine_region(sample_point, spline_top, spline_bottom)
                
                if sample_region in ['top_absorber', 'bottom_absorber']:
                    dt = t_samples[i+1] - t_samples[i] if i > 0 else t_sample
                    absorbed_intensity = intensity * alpha * dt * np.exp(-alpha * t_sample)
                    
                    if absorbed_intensity > 1e-8:
                        absorbed_points.append([sample_point[0], sample_point[1], absorbed_intensity])
            
            intensity *= np.exp(-alpha * t)
        
        # Handle different intersection types (same as before)
        if intersection_type == 'top_interface':
            # [Keep existing code for top_interface handling]
            x_int = intersection_point[0]
            normal = get_surface_normal(spline_top, x_int)
            
            if current_region == 'top_waveguide':
                n1, n2 = n_wg, n_abs
            else:
                n1, n2 = n_abs, n_wg
            
            cos_theta_i = -np.dot(direction, normal)
            theta_i = np.arccos(abs(cos_theta_i))
            
            R, T, theta_t = fresnel_coefficients(theta_i, n1, n2)
            
            if np.random.random() < R or theta_t is None:
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-8 * reflected_direction, reflected_direction, intensity]
            else:
                cos_theta_t = np.cos(theta_t)
                normal_component = np.dot(direction, normal)
                tangent = direction - normal_component * normal
                tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0])
                
                transmitted_direction = (n1/n2) * tangent + np.sign(normal_component) * cos_theta_t * normal
                transmitted_direction = transmitted_direction / np.linalg.norm(transmitted_direction)
                
                current_ray = [intersection_point + 1e-6 * transmitted_direction, transmitted_direction, intensity * T]
        
        elif intersection_type == 'bottom_interface':
            # [Keep existing code for bottom_interface handling]
            x_int = intersection_point[0]
            normal = get_surface_normal(spline_bottom, x_int)
            
            if current_region == 'bottom_waveguide':
                n1, n2 = n_wg, n_abs
            else:
                n1, n2 = n_abs, n_wg
            
            cos_theta_i = -np.dot(direction, normal)
            theta_i = np.arccos(abs(cos_theta_i))
            
            R, T, theta_t = fresnel_coefficients(theta_i, n1, n2)
            
            if np.random.random() < R or theta_t is None:
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
            else:
                cos_theta_t = np.cos(theta_t)
                normal_component = np.dot(direction, normal)
                tangent = direction - normal_component * normal
                tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0])
                
                transmitted_direction = (n1/n2) * tangent + np.sign(normal_component) * cos_theta_t * normal
                transmitted_direction = transmitted_direction / np.linalg.norm(transmitted_direction)
                
                current_ray = [intersection_point + 1e-6 * transmitted_direction, transmitted_direction, intensity * T]
        
        elif intersection_type == 'layer_interface':
            # [Keep existing layer_interface code]
            x_int = intersection_point[0]
            
            if current_region in ['top_waveguide', 'top_absorber']:
                if x_int <= rectangle_width:
                    if spline_bottom(x_int) < rectangle_height:
                        if current_region == 'top_waveguide':
                            n1, n2 = n_wg, n_abs
                        else:
                            n1, n2 = n_abs, n_abs
                    else:
                        if current_region == 'top_waveguide':
                            n1, n2 = n_wg, n_wg
                        else:
                            n1, n2 = n_abs, n_wg
                else:
                    n1, n2 = n_wg, n_abs
            else:
                if x_int <= rectangle_width:
                    if spline_top(x_int) + rectangle_height > rectangle_height:
                        if current_region == 'bottom_waveguide':
                            n1, n2 = n_wg, n_abs
                        else:
                            n1, n2 = n_abs, n_abs
                    else:
                        if current_region == 'bottom_waveguide':
                            n1, n2 = n_wg, n_wg
                        else:
                            n1, n2 = n_abs, n_wg
                else:
                    n1, n2 = n_wg, n_abs
            
            normal = np.array([0, 1]) if direction[1] > 0 else np.array([0, -1])
            cos_theta_i = abs(np.dot(direction, normal))
            theta_i = np.arccos(cos_theta_i) if cos_theta_i <= 1 else 0
            
            R, T, theta_t = fresnel_coefficients(theta_i, n1, n2)
            
            if np.random.random() < R or theta_t is None:
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
            else:
                cos_theta_t = np.cos(theta_t)
                normal_component = np.dot(direction, normal)
                tangent = direction - normal_component * normal
                tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0])
                
                transmitted_direction = (n1/n2) * tangent + np.sign(normal_component) * cos_theta_t * normal
                transmitted_direction = transmitted_direction / np.linalg.norm(transmitted_direction)
                
                current_ray = [intersection_point + 1e-6 * transmitted_direction, transmitted_direction, intensity * T]
        
        else:
            break
        
        if intensity < 1e-6:
            break
    
    return absorbed_points, ray_paths

# Main simulation
print("Creating dual-layer system...")
spline_top, x_control, y_control = create_interface_curve()
spline_bottom = spline_top  # Same interface shape for both layers

absorbed_energy_map = []
all_ray_paths = []

print("Starting dual-layer ray simulation...")

start_t = time.time()

rays = generate_rays_customized(num_rays)
for i, ray in enumerate(rays):
    if i % 100 == 0:
        print(f"Processing ray {i}/{num_rays}")
    absorbed_points, ray_paths = simulate_dual_layer_ray_with_paths(ray, spline_top, spline_bottom)
    absorbed_energy_map.extend(absorbed_points)
    all_ray_paths.extend(ray_paths)

print(f"Simulation complete. Found {len(absorbed_energy_map)} absorption events.")

# Visualization
if absorbed_energy_map:
    absorbed_data = np.array(absorbed_energy_map)
    x_absorbed = absorbed_data[:, 0]
    y_absorbed = absorbed_data[:, 1]
    intensity_absorbed = absorbed_data[:, 2]
else:
    x_absorbed = np.array([])
    y_absorbed = np.array([])
    intensity_absorbed = np.array([])

plt.figure(figsize=(12, 10))

# Print statistics
print(f"\n=== DUAL-LAYER SYSTEM STATISTICS ===")
print(f"Total rays: {num_rays}")
print(f"Absorption events: {len(absorbed_energy_map)}")
if len(absorbed_energy_map) > 0:
    print(f"Total absorbed energy: {np.sum(intensity_absorbed):.3f}")
    print(f"Average absorption per event: {np.mean(intensity_absorbed):.3f}")
    
    # Analyze absorption by layer
    top_absorption = np.sum(intensity_absorbed[y_absorbed > rectangle_height + layer_gap])
    bottom_absorption = np.sum(intensity_absorbed[y_absorbed < rectangle_height])
    
    print(f"Top layer absorption: {top_absorption:.3f} ({top_absorption/np.sum(intensity_absorbed)*100:.1f}%)")
    print(f"Bottom layer absorption: {bottom_absorption:.3f} ({bottom_absorption/np.sum(intensity_absorbed)*100:.1f}%)")

# Plot 1: Dual-layer geometry
plt.subplot(2, 2, 1)
x_curve = np.linspace(0, rectangle_width, 200)
y_curve_top = spline_top(x_curve)
y_curve_bottom = spline_bottom(x_curve)

# Draw geometry
plt.fill_between(x_curve, y_curve_top + rectangle_height + layer_gap, 
                total_height, alpha=0.3, color='orange', label='Top Absorber')
plt.fill_between(x_curve, rectangle_height + layer_gap, 
                y_curve_top + rectangle_height + layer_gap, alpha=0.3, color='lightblue', label='Top Waveguide')

# Gap
plt.fill_between([0, rectangle_width], rectangle_height, rectangle_height + layer_gap, 
                alpha=0.2, color='lightgray', label='Air Gap')

# Bottom layer  
plt.fill_between(x_curve, y_curve_bottom, rectangle_height, alpha=0.3, color='red', label='Bottom Absorber')
plt.fill_between(x_curve, 0, y_curve_bottom, alpha=0.3, color='cyan', label='Bottom Waveguide')

# Draw ray paths
print(f"Drawing {len(all_ray_paths)} ray segments...")

# Sample a subset of rays for visualization (to avoid overcrowding)
max_rays_to_show = 200  # Adjust this number
ray_subset = all_ray_paths[::max(1, len(all_ray_paths)//max_rays_to_show)]

for i, path in enumerate(ray_subset):
    start = path['start']
    end = path['end']
    region = path['region']
    intensity = path['intensity']
    bounce = path['bounce']
    
    # Color-code by region and intensity
    if region in ['top_waveguide', 'bottom_waveguide']:
        color = 'blue'
        alpha = min(0.8, intensity)
        linewidth = 1.5
    elif region in ['top_absorber', 'bottom_absorber']:
        color = 'red'
        alpha = min(0.6, intensity)
        linewidth = 1.0
    else:
        color = 'gray'
        alpha = 0.3
        linewidth = 0.5
    
    # Draw ray segment
    plt.plot([start[0], end[0]], [start[1], end[1]], 
             color=color, alpha=alpha, linewidth=linewidth)

# Interface curves
plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'b-', linewidth=2, label='Top Interface')
plt.plot(x_curve, y_curve_bottom, 'g-', linewidth=2, label='Bottom Interface')

plt.xlim(0, rectangle_width)
plt.ylim(0, total_height)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Dual-Layer System with Ray Paths')
plt.grid(True, alpha=0.3)



# Replace the Plot 2 section (around line 250):
plt.subplot(2, 2, 2)
if len(x_absorbed) > 0:
    # MUCH higher resolution
    bins_x = 100  # Was 25
    bins_y = 100  # Was 30
    
    H, xedges, yedges = np.histogram2d(x_absorbed, y_absorbed, 
                                       bins=[bins_x, bins_y],  # 100x100 = 10,000 bins
                                       range=[[0, rectangle_width], [0, total_height]],
                                       weights=intensity_absorbed)
    
    # ADJUST BRIGHTNESS/COLOR SCALE HERE
    # Option 1: Manual brightness control
    max_brightness = np.max(H) * 1.0 
    min_brightness = 0  # Set minimum
    
    # Option 2: Percentile-based (recommended for better contrast)
    # max_brightness = np.percentile(H[H > 0], 95)  # 95th percentile as max
    # min_brightness = np.percentile(H[H > 0], 5)   # 5th percentile as min
    
    # Option 3: Fixed values
    # max_brightness = 0.01  # Set your desired maximum value
    # min_brightness = 0
    
    # Use better interpolation with brightness control
    im = plt.imshow(H.T, extent=[0, rectangle_width, 0, total_height], 
                   origin='lower', cmap='viridis', aspect='auto', 
                   interpolation='bicubic',
                   vmin=min_brightness, vmax=max_brightness)  # Control brightness here
    
    plt.colorbar(im, label='Absorbed intensity')
    
    # Overlay interface curves with better visibility
    plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'white', linewidth=3, alpha=0.8)
    plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'red', linewidth=2)
    plt.plot(x_curve, y_curve_bottom, 'white', linewidth=3, alpha=0.8)
    plt.plot(x_curve, y_curve_bottom, 'red', linewidth=2)
    
    # Debug info
    print(f"Histogram max value: {np.max(H):.6f}")
    print(f"Color scale: {min_brightness:.6f} to {max_brightness:.6f}")
    
else:
    plt.text(0.5, 0.5, 'No absorption data', ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=12)

plt.title('2D Absorption Distribution')

# Plot 3: 1D distribution along x-axis
plt.subplot(2, 2, 3)
if len(x_absorbed) > 0:
    bins = np.linspace(0, rectangle_width, 20)
    bin_energy, _ = np.histogram(x_absorbed, bins=bins, weights=intensity_absorbed)
    
    plt.bar((bins[:-1] + bins[1:]) / 2, bin_energy, width=0.04, align='center')
    plt.xlabel("x position [m]")
    plt.ylabel("Total absorbed intensity")
    plt.title("X-axis Absorption Distribution")
    plt.grid(True)

# Plot 4: 1D distribution along y-axis
plt.subplot(2, 2, 4)
if len(y_absorbed) > 0:
    bins_y = np.linspace(0, total_height, 25)
    bin_energy_y, _ = np.histogram(y_absorbed, bins=bins_y, weights=intensity_absorbed)
    
    plt.barh((bins_y[:-1] + bins_y[1:]) / 2, bin_energy_y, height=total_height/25, align='center')
    plt.ylabel("y position [m]")
    plt.xlabel("Total absorbed intensity")
    plt.title("Y-axis Absorption Distribution")
    plt.grid(True)

# Print timing results
print(f"Took {time.time() - start_t}s.")

plt.tight_layout()
plt.show()




