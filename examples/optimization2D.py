import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# parameters
n_wg = 1.4             # waveguide refractive index
n_abs = 1.6            # absorber refractive index (same as outside)
alpha = 1.0            # absorption coefficient
rectangle_width = 1.0  
rectangle_height = 0.5
num_rays = 300        # number of rays

# define interface curve using cubic spline with 20 control points

def create_interface_curve():
    # Create 20 control points for the interface
    x_control = np.linspace(0, rectangle_width, 20)
    # Initialize with diagonal line, can be modified for optimization
    y_control = rectangle_height - x_control * 0.5
    # Ensure boundary conditions
    y_control[0] = rectangle_height
    y_control[-1] = 0
    
    # Create cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')
    return spline, x_control, y_control

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
"""
def create_interface_curve():
    # Create 20 control points for the interface
    x_control = np.linspace(0, rectangle_width, 20)
    
    # Adjustable parabolic curve
    # y = height * (1 - (x/width)^power)
    
    power = 2.0  # 2.0 = parabola, higher = steeper fall, lower = gentler fall
    
    x_normalized = x_control / rectangle_width  # Normalize to [0, 1]
    
    # Parabolic fall with adjustable steepness
    y_control = rectangle_height * (1 - x_normalized**power)
    
    # Ensure exact boundary conditions
    y_control[0] = rectangle_height  # Start at top
    y_control[-1] = 0                # End at bottom
    
    # Create cubic spline
    spline = CubicSpline(x_control, y_control, bc_type='natural')
    return spline, x_control, y_control
"""
    
# Fresnel reflection and transmission coefficients
def fresnel_coefficients(theta_i, n1, n2):
    """Calculate Fresnel coefficients for s-polarized light"""
    cos_i = np.cos(theta_i)
    sin_i = np.sin(theta_i)
    
    # Check for total internal reflection
    sin_t_squared = (n1/n2)**2 * sin_i**2
    if sin_t_squared > 1:
        return 1.0, 0.0, None  # Total internal reflection
    
    cos_t = np.sqrt(1 - sin_t_squared)
    
    # Fresnel equations for s-polarization
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    t_s = 2*n1*cos_i / (n1*cos_i + n2*cos_t)
    
    R = r_s**2
    T = (n2*cos_t)/(n1*cos_i) * t_s**2
    
    theta_t = np.arcsin((n1/n2) * sin_i)
    
    return R, T, theta_t

# Lambertian distribution
def lambertian_sample_theta(n, theta_max=np.pi/2):
    p = np.random.uniform(0, 1, n)
    theta = np.arcsin(np.sqrt(p) * np.sin(theta_max))
    return theta

# emit rays from left boundary
def generate_rays(n):
    thetas = lambertian_sample_theta(n)
    rays = []
    for theta in thetas:
        origin = np.array([0.0, np.random.uniform(0, rectangle_height)])
        direction = np.array([np.cos(theta), np.sin(theta)])
        intensity = 1.0
        rays.append([origin, direction, intensity])
    return rays

# calculate surface normal at intersection point
def get_surface_normal(spline, x):
    dy_dx = spline.derivative()(x)
    normal = np.array([-dy_dx, 1])
    normal = normal / np.linalg.norm(normal)
    return normal

# intersect with interface curve - improved version
def intersect_interface(ray, spline):
    origin, direction, _ = ray
    
    # Parametric ray equation: P = origin + t * direction
    def intersection_equation(t):
        if t <= 1e-6:  # Small positive value to avoid starting point
            return float('inf')
        point = origin + t * direction
        x, y = point
        if x < 0 or x > rectangle_width:
            return float('inf')
        try:
            spline_y = spline(x)
            return y - spline_y
        except:
            return float('inf')
    
    # More robust intersection finding
    t_max = 10.0  # Maximum parameter value to search
    dt = 0.01     # Step size for initial search
    
    # Find sign changes more carefully
    t_current = dt
    prev_val = intersection_equation(t_current)
    
    while t_current < t_max:
        t_current += dt
        current_val = intersection_equation(t_current)
        
        # Check for valid values and sign change
        if (not np.isinf(prev_val) and not np.isinf(current_val) and 
            not np.isnan(prev_val) and not np.isnan(current_val) and
            prev_val * current_val < 0):
            
            try:
                # Use fsolve to find exact intersection
                t_intersect = fsolve(intersection_equation, t_current - dt/2)[0]
                if t_intersect > 1e-6:
                    intersection_point = origin + t_intersect * direction
                    x_int, y_int = intersection_point
                    if (0 <= x_int <= rectangle_width and 
                        0 <= y_int <= rectangle_height):
                        return intersection_point, t_intersect
            except:
                pass
        
        prev_val = current_val
    
    return None, None

# intersect with bottom boundary
def intersect_bottom(ray):
    origin, direction, _ = ray
    if direction[1] >= 0:  # Ray going up, won't hit bottom
        return None, None
    
    # y = 0, solve for t: origin[1] + t * direction[1] = 0
    t = -origin[1] / direction[1]
    if t > 1e-6:  # Small positive value to avoid numerical issues
        intersection_point = origin + t * direction
        x_int = intersection_point[0]
        if 0 <= x_int <= rectangle_width:
            return intersection_point, t
    
    return None, None

# intersect with top and right boundaries
def intersect_boundaries(ray):
    origin, direction, _ = ray
    intersections = []
    
    # Top boundary (y = rectangle_height)
    if direction[1] > 0:
        t = (rectangle_height - origin[1]) / direction[1]
        if t > 1e-6:
            x_int = origin[0] + t * direction[0]
            if 0 <= x_int <= rectangle_width:
                intersections.append((np.array([x_int, rectangle_height]), t, 'top'))
    
    # Right boundary (x = rectangle_width)
    if direction[0] > 0:
        t = (rectangle_width - origin[0]) / direction[0]
        if t > 1e-6:
            y_int = origin[1] + t * direction[1]
            if 0 <= y_int <= rectangle_height:
                intersections.append((np.array([rectangle_width, y_int]), t, 'right'))
    
    # Return closest intersection
    if intersections:
        intersections.sort(key=lambda x: x[1])  # Sort by parameter t
        return intersections[0][0], intersections[0][1]
    
    return None, None

# simulate ray propagation with Fresnel effects - CORRECTED VERSION
def simulate_ray(ray, spline, max_bounces=50):
    absorbed_points = []
    current_ray = ray.copy()
    
    for bounce in range(max_bounces):
        origin, direction, intensity = current_ray
        
        # Find all possible intersections
        int_point, t_int = intersect_interface(current_ray, spline)
        bottom_point, t_bottom = intersect_bottom(current_ray)
        boundary_point, t_boundary = intersect_boundaries(current_ray)
        
        # Determine which intersection occurs first
        intersections = []
        if int_point is not None:
            intersections.append((int_point, t_int, 'interface'))
        if bottom_point is not None:
            intersections.append((bottom_point, t_bottom, 'bottom'))
        if boundary_point is not None:
            intersections.append((boundary_point, t_boundary, 'boundary'))
        
        if not intersections:
            break  # No intersections found, ray exits
        
        # Sort by parameter t and take the closest
        intersections.sort(key=lambda x: x[1])
        intersection_point, t, intersection_type = intersections[0]
        
        # CORRECTED ABSORPTION SIMULATION
        # Sample absorption along the ray path, not just at intersection
        current_x = origin[0]
        try:
            if origin[1] > spline(current_x):  # Starting in absorber
                # Sample multiple points along the ray path for absorption
                num_samples = max(10, int(t * 100))  # More samples for longer paths
                t_samples = np.linspace(0, t, num_samples)
                
                for i, t_sample in enumerate(t_samples[1:]):  # Skip origin
                    sample_point = origin + t_sample * direction
                    
                    # Check if this point is still in absorber
                    if (0 <= sample_point[0] <= rectangle_width and 
                        0 <= sample_point[1] <= rectangle_height and
                        sample_point[1] > spline(sample_point[0])):
                        
                        # Calculate absorption for this segment
                        dt = t_samples[i+1] - t_samples[i] if i > 0 else t_sample
                        absorbed_intensity = intensity * alpha * dt * np.exp(-alpha * t_sample)
                        
                        if absorbed_intensity > 1e-8:  # Record significant absorption
                            absorbed_points.append([sample_point[0], sample_point[1], absorbed_intensity])
                
                # Update ray intensity after passing through absorber
                intensity *= np.exp(-alpha * t)
        except:
            pass
        
        # Handle different intersection types (same as before)
        if intersection_type == 'interface':
            # Handle Fresnel reflection/transmission at interface
            x_int = intersection_point[0]
            normal = get_surface_normal(spline, x_int)
            
            # Determine incident medium
            try:
                if origin[1] < spline(origin[0]):  # Coming from waveguide
                    n1, n2 = n_wg, n_abs
                else:  # Coming from absorber
                    n1, n2 = n_abs, n_wg
            except:
                n1, n2 = n_wg, n_abs  # Default assumption
            
            # Calculate incident angle
            cos_theta_i = -np.dot(direction, normal)
            if abs(cos_theta_i) > 1:
                cos_theta_i = np.sign(cos_theta_i)
            theta_i = np.arccos(abs(cos_theta_i))
            
            # Get Fresnel coefficients
            R, T, theta_t = fresnel_coefficients(theta_i, n1, n2)
            
            # Determine if ray reflects or transmits
            if np.random.random() < R:
                # Reflection
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
            else:
                # Transmission
                if theta_t is not None:
                    # Calculate transmitted direction using Snell's law
                    cos_theta_t = np.cos(theta_t)
                    # Project direction onto normal and tangent
                    normal_component = np.dot(direction, normal)
                    tangent = direction - normal_component * normal
                    tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0])
                    
                    transmitted_direction = (n1/n2) * tangent + np.sign(normal_component) * cos_theta_t * normal
                    transmitted_direction = transmitted_direction / np.linalg.norm(transmitted_direction)
                    
                    current_ray = [intersection_point + 1e-6 * transmitted_direction, transmitted_direction, intensity * T]
                else:
                    # Total internal reflection
                    reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                    current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
        
        elif intersection_type == 'bottom':
            # Handle reflection/transmission at bottom boundary
            normal = np.array([0, 1])  # Normal pointing up
            n1, n2 = n_wg, n_abs
            
            cos_theta_i = -np.dot(direction, normal)
            theta_i = np.arccos(abs(cos_theta_i))
            
            R, T, theta_t = fresnel_coefficients(theta_i, n1, n2)
            
            if np.random.random() < R:
                # Reflection
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
            else:
                # Transmission (ray exits the domain)
                break
        
        else:  # boundary (top or right)
            # Ray exits the domain
            break
        
        # Safety check for intensity
        if intensity < 1e-6:
            break
    
    return absorbed_points

# Main simulation
spline, x_control, y_control = create_interface_curve()
absorbed_energy_map = []

print("Starting ray simulation...")
rays = generate_rays(num_rays)
for i, ray in enumerate(rays):
    if i % 100 == 0:
        print(f"Processing ray {i}/{num_rays}")
    absorbed_points = simulate_ray(ray, spline)
    absorbed_energy_map.extend(absorbed_points)

print(f"Simulation complete. Found {len(absorbed_energy_map)} absorption events.")

# Convert to numpy array for easier handling
if absorbed_energy_map:
    absorbed_data = np.array(absorbed_energy_map)
    x_absorbed = absorbed_data[:, 0]
    y_absorbed = absorbed_data[:, 1]
    intensity_absorbed = absorbed_data[:, 2]
else:
    x_absorbed = np.array([])
    y_absorbed = np.array([])
    intensity_absorbed = np.array([])

# 2D absorption distribution visualization
plt.figure(figsize=(12, 8))

# Plot 1: Interface curve and control points
plt.subplot(2, 2, 1)
x_curve = np.linspace(0, rectangle_width, 200)
y_curve = spline(x_curve)
plt.plot(x_curve, y_curve, 'b-', linewidth=2, label='Interface curve')
plt.scatter(x_control, y_control, c='red', s=50, zorder=5, label='Control points')
plt.fill_between(x_curve, y_curve, rectangle_height, alpha=0.3, color='orange', label='Absorber')
plt.fill_between(x_curve, 0, y_curve, alpha=0.3, color='lightblue', label='Waveguide')
plt.xlim(0, rectangle_width)
plt.ylim(0, rectangle_height)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Interface Curve')
plt.legend()
plt.grid(True)

# Plot 2: 2D contour plot of absorption
# Plot 2: 2D contour plot of absorption
plt.subplot(2, 2, 2)
if len(x_absorbed) > 0:
    # Create 2D histogram with explicit range covering the full domain
    bins_x = 25
    bins_y = 25
    H, xedges, yedges = np.histogram2d(x_absorbed, y_absorbed, 
                                       bins=[bins_x, bins_y],
                                       range=[[0, rectangle_width], [0, rectangle_height]],
                                       weights=intensity_absorbed)
    
    # Use imshow for perfect coverage of the domain
    im = plt.imshow(H.T, extent=[0, rectangle_width, 0, rectangle_height], 
                   origin='lower', cmap='viridis', aspect='auto', interpolation='bilinear')
    plt.colorbar(im, label='Absorbed intensity')
    
    # Overlay interface curve with better visibility
    plt.plot(x_curve, y_curve, 'white', linewidth=3, alpha=0.8)
    plt.plot(x_curve, y_curve, 'red', linewidth=2, alpha=1.0)
else:
    # Create empty plot with proper background
    plt.imshow(np.zeros((20, 20)), extent=[0, rectangle_width, 0, rectangle_height], 
              origin='lower', cmap='viridis', aspect='auto', alpha=0.3)
    plt.colorbar(label='Absorbed intensity')
    plt.text(0.5, 0.5, 'No absorption data', ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.xlim(0, rectangle_width)
plt.ylim(0, rectangle_height)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('2D Absorption Distribution')

# Plot 3: 1D distribution along x-axis
plt.subplot(2, 2, 3)
if len(x_absorbed) > 0:
    bins = np.linspace(0, rectangle_width, 20)
    bin_energy, _ = np.histogram(x_absorbed, bins=bins, weights=intensity_absorbed)
    
    plt.bar((bins[:-1] + bins[1:]) / 2, bin_energy, width=0.04, align='center')
    plt.xlabel("x position [m]")
    plt.ylabel("Total absorbed intensity")
    plt.title("1D Absorption Distribution (x-axis)")
    plt.grid(True)

# Plot 4: 1D distribution along y-axis
plt.subplot(2, 2, 4)
if len(y_absorbed) > 0:
    bins_y = np.linspace(0, rectangle_height, 15)
    bin_energy_y, _ = np.histogram(y_absorbed, bins=bins_y, weights=intensity_absorbed)
    
    plt.barh((bins_y[:-1] + bins_y[1:]) / 2, bin_energy_y, height=0.02, align='center')
    plt.ylabel("y position [m]")
    plt.xlabel("Total absorbed intensity")
    plt.title("1D Absorption Distribution (y-axis)")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print statistics
print(f"Total rays: {num_rays}")
print(f"Absorption events: {len(absorbed_energy_map)}")
if len(absorbed_energy_map) > 0:
    print(f"Total absorbed energy: {np.sum(intensity_absorbed):.3f}")
    print(f"Average absorption per event: {np.mean(intensity_absorbed):.3f}")