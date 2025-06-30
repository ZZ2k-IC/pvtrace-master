#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# Parameters
n_wg = 1.4585             # waveguide refractive index
n_abs = 1.6               # absorber refractive index
alpha = 0.1             # absorption coefficient
rectangle_width = 15.0  
rectangle_height = 0.5
layer_gap = 0.01          # Gap between top and bottom layers
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

def lambertian_sample_theta(n, theta_max=np.pi/3):
    p = np.random.uniform(0, 1, n)
    theta = np.arcsin(np.sqrt(p) * np.sin(theta_max))
    return theta

def generate_rays(n):
    """Generate rays in TOP layer (y > rectangle_height + layer_gap)"""
    thetas = lambertian_sample_theta(n)
    rays = []
    for theta in thetas:
        # Start rays in top layer
        y_start = rectangle_height + layer_gap + np.random.uniform(0, rectangle_height)
        origin = np.array([0.0, y_start])
        direction = np.array([np.cos(theta), np.sin(theta)])
        intensity = 1.0
        rays.append([origin, direction, intensity])
    return rays

def get_surface_normal(spline, x):
    dy_dx = spline.derivative()(x)
    normal = np.array([-dy_dx, 1])
    normal = normal / np.linalg.norm(normal)
    return normal

def determine_region(point, spline_top, spline_bottom):
    """Determine which region a point is in"""
    x, y = point
    
    if y > rectangle_height + layer_gap:
        # Top layer
        if y > spline_top(x) + rectangle_height + layer_gap:
            return 'top_absorber'
        else:
            return 'top_waveguide'
    elif y < rectangle_height:
        # Bottom layer  
        if y > spline_bottom(x):
            return 'bottom_absorber'
        else:
            return 'bottom_waveguide'
    else:
        # Gap region (air)
        return 'gap'

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
    
    t_max = 10.0
    dt = 0.01
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

def intersect_layer_boundaries(ray):
    """Find intersections with layer boundaries"""
    origin, direction, _ = ray
    intersections = []
    
    # Layer boundaries
    boundaries = [
        (0, 'bottom_waveguide'),                           # y = 0
        (rectangle_height, 'bottom_top'),                   # y = 0.5
        (rectangle_height + layer_gap, 'gap_top'),          # y = 0.55  
        (total_height, 'top_boundary')                      # y = 1.05
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

def simulate_dual_layer_ray(ray, spline_top, spline_bottom, max_bounces=100):
    """Simulate ray in dual-layer system"""
    absorbed_points = []
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
        
        # Handle absorption along path
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
        
        # Handle different intersection types
        if intersection_type == 'top_interface':
            # Handle top layer interface
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
                # Reflection
                reflected_direction = direction - 2 * np.dot(direction, normal) * normal
                current_ray = [intersection_point + 1e-6 * reflected_direction, reflected_direction, intensity]
            else:
                # Transmission
                cos_theta_t = np.cos(theta_t)
                normal_component = np.dot(direction, normal)
                tangent = direction - normal_component * normal
                tangent = tangent / np.linalg.norm(tangent) if np.linalg.norm(tangent) > 0 else np.array([1, 0])
                
                transmitted_direction = (n1/n2) * tangent + np.sign(normal_component) * cos_theta_t * normal
                transmitted_direction = transmitted_direction / np.linalg.norm(transmitted_direction)
                
                current_ray = [intersection_point + 1e-6 * transmitted_direction, transmitted_direction, intensity * T]
        
        elif intersection_type == 'bottom_interface':
            # Handle bottom layer interface (same logic as top)
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
        
        elif intersection_type in ['gap_top', 'bottom_top']:
            # Ray crossing between layers through gap - just continue propagating
            current_ray = [intersection_point + 1e-6 * direction, direction, intensity]
        
        else:
            # Ray hits boundary - exit simulation
            break
        
        if intensity < 1e-6:
            break
    
    return absorbed_points

# Main simulation
print("Creating dual-layer system...")
spline_top, x_control, y_control = create_interface_curve()
spline_bottom = spline_top  # Same interface shape for both layers

absorbed_energy_map = []

print("Starting dual-layer ray simulation...")
rays = generate_rays(num_rays)
for i, ray in enumerate(rays):
    if i % 100 == 0:
        print(f"Processing ray {i}/{num_rays}")
    absorbed_points = simulate_dual_layer_ray(ray, spline_top, spline_bottom)
    absorbed_energy_map.extend(absorbed_points)

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

# Plot 1: Dual-layer geometry
plt.subplot(2, 2, 1)
x_curve = np.linspace(0, rectangle_width, 200)
y_curve_top = spline_top(x_curve)
y_curve_bottom = spline_bottom(x_curve)

# Top layer
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

# Interface curves
plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'b-', linewidth=2, label='Top Interface')
plt.plot(x_curve, y_curve_bottom, 'g-', linewidth=2, label='Bottom Interface')

plt.xlim(0, rectangle_width)
plt.ylim(0, total_height)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Dual-Layer System Geometry')
plt.legend()
plt.grid(True)

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
    
    # Use better interpolation
    im = plt.imshow(H.T, extent=[0, rectangle_width, 0, total_height], 
                   origin='lower', cmap='viridis', aspect='auto', 
                   interpolation='bicubic')  # Better than default
    plt.colorbar(im, label='Absorbed intensity')
    
    # Overlay interface curves with better visibility
    plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'white', linewidth=3, alpha=0.8)
    plt.plot(x_curve, y_curve_top + rectangle_height + layer_gap, 'red', linewidth=2)
    plt.plot(x_curve, y_curve_bottom, 'white', linewidth=3, alpha=0.8)
    plt.plot(x_curve, y_curve_bottom, 'red', linewidth=2)
    
else:
    plt.text(0.5, 0.5, 'No absorption data', ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=12)

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

plt.tight_layout()
plt.show()

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
