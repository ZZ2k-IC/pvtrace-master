from pvtrace import *
import time
import functools
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# unit is cm

# Set up the rays number
rays_num = 40000

# Add nodes to the scene graph
world = Node(
    name="World",
    geometry=Sphere(
        radius=9.0,
        material=Material(refractive_index=1.0),
    )
)

# box = Node(
#     name="Santovac 5",
#     geometry=Box(
#         (1, 1, 0.2),
#         material=Material(refractive_index=1.63),
#     ),
#     parent=world
# )


horn = Node(
   name = "Waveguide",
   geometry = Mesh(
       trimesh = trimesh.load(r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\STL_file\pyramid.stl"),
       material = Material(
           refractive_index = 2.32,
       ),
   ),
   parent = world
)
horn.translate((0, 0, 0.54))


# cylinder = Node(
#     name = "Waveguide",
#     geometry = 
#     Cylinder(
#         radius=0.25,
#         length=6.2,
#         material=Material(refractive_index=1.45)),
#         parent = world
# )
# cylinder.translate((0, 0, 3.0))  # Position cylinder at z=6.0

# Prism = Node(
#     name="Glass Prism Waveguide",
#     geometry=Box(
#         (0.5, 0.5, 6.2),
#         material=Material(refractive_index=1.45),
#     ),
#     parent=world
# )
# Prism.translate((0, 0, 3.0))

# Add source of photons
light = Node(
    name="Light (555nm)",
    parent=world,
    light=Light(
        position=functools.partial(rectangular_mask, 0.16, 0.13), # Rectangular mask of size 0.32cm x 0.26cm
        direction=functools.partial(lambertian, np.pi*25.5/180) # Maximum beam angle is ~43 degrees.
    )
)


# Add detector at bottom of cylinder (z=0.1) - detects rays coming from above
bottom_detector = create_planar_detector_node(
    name="Bottom Detector",
    length=0.3,  # Larger than cylinder radius (0.5) to catch all rays
    width=0.3,
    normal=(0, 0, 1),  # Normal pointing up
    detection_direction=(0, 0, 1),  # Detect rays coming from above (downward)
    parent=world
)
bottom_detector.translate((0, 0, 1.38))  # Position at cylinder bottom

# # Add detector at top of cylinder (z=3.2) - detects rays coming from below  
# top_detector = create_planar_detector_node(
#     name="Top Detector",
#     length=2.0,  # Larger than cylinder radius to catch all rays
#     width=2.0,
#     normal=(0, 0, 1),  # Normal pointing up
#     detection_direction=(0, 0, 10.0),  # Detect rays coming from below (upward)
#     parent=world
# )
# top_detector.translate((0, 0, 6.2))  # Position at cylinder top

# Use meshcat to render the scene (optional)
viewer = MeshcatRenderer(open_browser=True, transparency=False, opacity=0.5, wireframe=True)
scene = Scene(world)
viewer.render(scene)


print("Starting simulation...")
start_t = time.time()  # Start timing here

# List to store initial ray directions for comparison
initial_ray_directions = []

for ray in scene.emit(rays_num):  # Note: using 'viewer' not 'scene'
    # Store initial ray direction
    initial_ray_directions.append(ray.direction)

    steps = photon_tracer.follow(scene, ray)
    path, surface_info, events = zip(*steps)
    viewer.add_ray_path(path)  # Note: using 'viewer' not 'renderer'

# Print timing results
print(f"Took {time.time() - start_t}s.")

# Print detection results
print(f"\nDetection Results:")
print(f"Bottom detector: {bottom_detector.detector_delegate.detected_count} rays detected")
# print(f"Top detector: {top_detector.detector_delegate.detected_count} rays detected")
# print(f"Total rays detected: {bottom_detector.detector_delegate.detected_count + top_detector.detector_delegate.detected_count}")

# Show detection efficiency
bottom_efficiency = bottom_detector.detector_delegate.detected_count / rays_num * 100
# top_efficiency = top_detector.detector_delegate.detected_count / rays_num * 100
print(f"Bottom detection efficiency: {bottom_efficiency:.1f}%")
# print(f"Top detection efficiency: {top_efficiency:.1f}%")


# After the simulation is complete, extract all detected ray directions
all_detected_rays = (bottom_detector.detector_delegate.detected_rays)

# Extract 3D direction vectors
detected_directions = []
for ray_info in all_detected_rays:
    direction = ray_info['direction']  # This is already a 3D vector [x, y, z]
    detected_directions.append(direction)

# Convert to numpy array for easier manipulation
detected_directions = np.array(detected_directions)

# Save to file
np.save('detected_ray_directions_pyramid.npy', detected_directions)

# Calculate azimuthal angles for detected rays
detected_azimuthal_angles = []
for ray_info in all_detected_rays:
    direction = ray_info['direction']
    # Calculate azimuthal angle (φ) from x and y components
    phi_rad = np.arctan2(direction[1], direction[0])
    phi_deg = np.degrees(phi_rad)
    # Convert to 0-360 range
    if phi_deg < 0:
        phi_deg += 360
    detected_azimuthal_angles.append(phi_deg)

# Calculate azimuthal angles for initial rays
initial_azimuthal_angles = []
for direction in initial_ray_directions:
    phi_rad = np.arctan2(direction[1], direction[0])
    phi_deg = np.degrees(phi_rad)
    # Convert to 0-360 range
    if phi_deg < 0:
        phi_deg += 360
    initial_azimuthal_angles.append(phi_deg)

# Create histogram with bins from 0 to 360 degrees
plt.figure(figsize=(12, 8))
bins_az = np.linspace(0, 360, 361)  # 360 bins, 1 degree each

# Plot detected rays histogram (solid bars)
counts_detected_az, bin_edges_az, patches_az = plt.hist(detected_azimuthal_angles, bins=bins_az, 
                                               edgecolor='black', alpha=0.7, 
                                               label='Detected Rays', color='blue')

# Plot initial rays histogram (dotted line)
counts_initial_az, _, _ = plt.hist(initial_azimuthal_angles, bins=bins_az, 
                                 histtype='step', linestyle='--', 
                                 linewidth=2, label='Initial Rays', color='red')

plt.xlabel('Azimuthal Angle (degrees)')
plt.ylabel('Number of Rays')
plt.title('Ray Num (Power) Distribution of Azimuthal Angles: Initial vs Detected Rays')
plt.grid(True, alpha=0.3)
plt.legend()

# Add bin labels for detected rays
bin_centers_az = (bin_edges_az[:-1] + bin_edges_az[1:]) / 2
for i, (center, count) in enumerate(zip(bin_centers_az, counts_detected_az)):
    if count > 0:
        plt.text(center, count + 0.1, f'{int(count)}', ha='center', va='bottom', color='blue')

# Add bin labels for initial rays (positioned above the step line)
for i, (center, count) in enumerate(zip(bin_centers_az, counts_initial_az)):
    if count > 0:
        # Position labels just above the step line height
        plt.text(center, count + 0.2, f'{int(count)}', ha='center', va='bottom', 
                color='red', style='italic', fontweight='bold')

plt.tight_layout()
plt.show()

# Generate the two lists for analysis
# 1. Mid-values of each bin
bin_mid_values_az = (bin_edges_az[:-1] + bin_edges_az[1:]) / 2

# 2. Relative heights (normalized so sum = 1)
# For detected rays
total_detected_az = np.sum(counts_detected_az)
detected_relative_heights_az = counts_detected_az / total_detected_az if total_detected_az > 0 else np.zeros_like(counts_detected_az)

# For initial rays  
total_initial_az = np.sum(counts_initial_az)
initial_relative_heights_az = counts_initial_az / total_initial_az if total_initial_az > 0 else np.zeros_like(counts_initial_az)

# Print azimuthal angle statistics
print(f"\nAzimuthal Angle Statistics:")
print(f"Initial rays: {len(initial_azimuthal_angles)} total")
print(f"  Mean azimuthal angle: {np.mean(initial_azimuthal_angles):.1f}°")
print(f"  Standard deviation: {np.std(initial_azimuthal_angles):.1f}°")
print(f"  Min angle: {np.min(initial_azimuthal_angles):.1f}°")
print(f"  Max angle: {np.max(initial_azimuthal_angles):.1f}°")

print(f"Detected rays: {len(detected_azimuthal_angles)} total")
print(f"  Mean azimuthal angle: {np.mean(detected_azimuthal_angles):.1f}°")
print(f"  Standard deviation: {np.std(detected_azimuthal_angles):.1f}°")
print(f"  Min angle: {np.min(detected_azimuthal_angles):.1f}°")
print(f"  Max angle: {np.max(detected_azimuthal_angles):.1f}°")

if all_detected_rays:
    # Calculate polar angles for detected rays
    detected_polar_angles = []
    for ray_info in all_detected_rays:
        direction = ray_info['direction']
        # Use absolute value of z-component for polar angle calculation
        cos_theta = abs(direction[2])
        # Clamp to valid range for arccos to avoid numerical errors
        cos_theta = np.clip(cos_theta, 0, 1)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        detected_polar_angles.append(theta_deg)
    
    # Calculate polar angles for initial rays
    initial_polar_angles = []
    for direction in initial_ray_directions:
        cos_theta = abs(direction[2])
        cos_theta = np.clip(cos_theta, 0, 1)
        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)
        initial_polar_angles.append(theta_deg)
    
    # Create histogram with 9 bins from 0 to 90 degrees
    plt.figure(figsize=(12, 8))
    bins = np.linspace(0, 90, 361)  # 10 edges = 9 bins
    
    # Plot detected rays histogram (solid bars)
    counts_detected, bin_edges, patches = plt.hist(detected_polar_angles, bins=bins, 
                                                   edgecolor='black', alpha=0.7, 
                                                   label='Detected Rays', color='blue')
    
    # Plot initial rays histogram (dotted line)
    counts_initial, _, _ = plt.hist(initial_polar_angles, bins=bins, 
                                   histtype='step', linestyle='--', 
                                   linewidth=2, label='Initial Rays', color='red')
    
    plt.xlabel('Polar Angle (degrees)')
    plt.ylabel('Number of Rays')
    plt.title('Ray Num (Power) Distribution of Polar Angles: Initial vs Detected Rays, solid angles are not the same for each bin')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add bin labels for detected rays
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for i, (center, count) in enumerate(zip(bin_centers, counts_detected)):
        if count > 0:
            plt.text(center, count + 0.1, f'{int(count)}', ha='center', va='bottom', color='blue')
    
    # Add bin labels for initial rays (positioned above the step line)
    for i, (center, count) in enumerate(zip(bin_centers, counts_initial)):
        if count > 0:
            # Position labels just above the step line height
            plt.text(center, count + 0.2, f'{int(count)}', ha='center', va='bottom', 
                    color='red', style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    # Generate the two lists for analysis
    # 1. Mid-values of each bin
    bin_mid_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Relative heights (normalized so sum = 1)
    # For detected rays
    total_detected = np.sum(counts_detected)
    detected_relative_heights = counts_detected / total_detected if total_detected > 0 else np.zeros_like(counts_detected)
    
    # For initial rays  
    total_initial = np.sum(counts_initial)
    initial_relative_heights = counts_initial / total_initial if total_initial > 0 else np.zeros_like(counts_initial)

    # Now create weighted histogram (1/sin(theta) weighting)
    # Calculate weights for detected rays
    detected_weights = []
    for angle_deg in detected_polar_angles:
        angle_rad = np.radians(angle_deg)
        # Avoid division by zero for angles very close to 0
        sin_theta = max(np.sin(angle_rad), 1e-10)
        weight = 1.0 / sin_theta
        detected_weights.append(weight)
    
    # Calculate weights for initial rays
    initial_weights = []
    for angle_deg in initial_polar_angles:
        angle_rad = np.radians(angle_deg)
        sin_theta = max(np.sin(angle_rad), 1e-10)
        weight = 1.0 / sin_theta
        initial_weights.append(weight)
    
    # Create weighted histogram
    plt.figure(figsize=(12, 8))
    
    # Plot weighted detected rays histogram
    counts_detected_weighted, bin_edges_weighted, patches_weighted = plt.hist(
        detected_polar_angles, bins=bins, weights=detected_weights,
        edgecolor='black', alpha=0.7, 
        label='Detected Rays (weighted)', color='blue')
    
    # Plot weighted initial rays histogram
    counts_initial_weighted, _, _ = plt.hist(
        initial_polar_angles, bins=bins, weights=initial_weights,
        histtype='step', linestyle='--', 
        linewidth=2, label='Initial Rays (weighted)', color='red')
    
    plt.xlabel('Polar Angle (degrees)')
    plt.ylabel('Weighted Count (1/sin θ)')
    plt.title('Weighted Distribution (Intensity) of Polar Angles: Initial vs Detected Rays, solid angles are the same for each bin')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add bin labels for weighted detected rays
    bin_centers = (bin_edges_weighted[:-1] + bin_edges_weighted[1:]) / 2
    for i, (center, count) in enumerate(zip(bin_centers, counts_detected_weighted)):
        if count > 0:
            plt.text(center, count + 0.05, f'{count:.1f}', ha='center', va='bottom', color='blue')
    
    # Add bin labels for weighted initial rays
    for i, (center, count) in enumerate(zip(bin_centers, counts_initial_weighted)):
        if count > 0:
            plt.text(center, count + 0.1, f'{count:.1f}', ha='center', va='bottom', 
                    color='red', style='italic', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    
    # Print angle statistics
    print(f"\nPolar Angle Statistics:")
    print(f"Initial rays: {len(initial_polar_angles)} total")
    print(f"  Mean polar angle: {np.mean(initial_polar_angles):.1f}°")
    print(f"  Standard deviation: {np.std(initial_polar_angles):.1f}°")
    print(f"  Min angle: {np.min(initial_polar_angles):.1f}°")
    print(f"  Max angle: {np.max(initial_polar_angles):.1f}°")
    
    print(f"Detected rays: {len(detected_polar_angles)} total")
    print(f"  Mean polar angle: {np.mean(detected_polar_angles):.1f}°")
    print(f"  Standard deviation: {np.std(detected_polar_angles):.1f}°")
    print(f"  Min angle: {np.min(detected_polar_angles):.1f}°")
    print(f"  Max angle: {np.max(detected_polar_angles):.1f}°")
    
else:
    print("No rays were detected!")

# Keep the script alive until Ctrl-C (optional)
print("\nCtrl-C to close")
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break