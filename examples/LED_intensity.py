from pvtrace import *
import time
import functools
import numpy as np
import trimesh
import matplotlib.pyplot as plt


# Set up the rays number
rays_num = 10000

# Add nodes to the scene graph
world = Node(
    name="World",
    geometry=Sphere(
        radius=9.0,
        material=Material(refractive_index=1.0),
    )
)
"""
box = Node(
    name="Santovac 5",
    geometry=Box(
        (1, 1, 0.2),
        material=Material(refractive_index=1.63),
    ),
    parent=world
)
"""

horn = Node(
    name = "Waveguide",
    geometry = Mesh(
        trimesh = trimesh.load(r"C:\Users\Zedd\Downloads\hornCY.stl"),
        material = Material(
            refractive_index = 1.45,
        ),
    ),
    parent = world
)
horn.translate((0, 0, 3.2))

"""
cylinder = Node(
    name = "Waveguide",
    geometry = 
    Cylinder(
        radius=0.25,
        length=6.2,
        material=Material(refractive_index=1.45)),
        parent = world
)
cylinder.translate((0, 0, 3.0))  # Position cylinder at z=6.0
"""

# Add source of photons
light = Node(
    name="Light (555nm)",
    parent=world,
    light=Light(
        position=functools.partial(rectangular_mask, 0.16, 0.13),
        direction=functools.partial(lambertian, np.pi*43/180) # Maximum beam angle is ~43 degrees.
    )
)


# Add detector at bottom of cylinder (z=0.1) - detects rays coming from above
bottom_detector = create_planar_detector_node(
    name="Bottom Detector",
    length=1.0,  # Larger than cylinder radius (0.5) to catch all rays
    width=1.0,
    normal=(0, 0, 1),  # Normal pointing up
    detection_direction=(0, 0, 1),  # Detect rays coming from above (downward)
    parent=world
)
bottom_detector.translate((0, 0, 6.0))  # Position at cylinder bottom

# Add detector at top of cylinder (z=3.2) - detects rays coming from below  
top_detector = create_planar_detector_node(
    name="Top Detector",
    length=2.0,  # Larger than cylinder radius to catch all rays
    width=2.0,
    normal=(0, 0, 1),  # Normal pointing up
    detection_direction=(0, 0, 10.0),  # Detect rays coming from below (upward)
    parent=world
)
top_detector.translate((0, 0, 6.2))  # Position at cylinder top

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
print(f"Top detector: {top_detector.detector_delegate.detected_count} rays detected")
print(f"Total rays detected: {bottom_detector.detector_delegate.detected_count + top_detector.detector_delegate.detected_count}")

# Show detection efficiency
bottom_efficiency = bottom_detector.detector_delegate.detected_count / rays_num * 100
top_efficiency = top_detector.detector_delegate.detected_count / rays_num * 100
print(f"Bottom detection efficiency: {bottom_efficiency:.1f}%")
print(f"Top detection efficiency: {top_efficiency:.1f}%")

all_detected_rays = (bottom_detector.detector_delegate.detected_rays + 
                    top_detector.detector_delegate.detected_rays)

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
    bins = np.linspace(0, 90, 19)  # 10 edges = 9 bins
    
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
    
    # Print histogram comparison
    print(f"\nHistogram Comparison:")
    print(f"{'Angle Range':<15} {'Initial':<10} {'Detected':<10} {'Efficiency':<12}")
    print("-" * 50)
    for i in range(len(counts_initial)):
        efficiency = (counts_detected[i] / counts_initial[i] * 100) if counts_initial[i] > 0 else 0
        print(f"{bin_edges[i]:.1f}°-{bin_edges[i+1]:.1f}°{'':<4} {int(counts_initial[i]):<10} {int(counts_detected[i]):<10} {efficiency:.1f}%")

else:
    print("No rays were detected!")

# Keep the script alive until Ctrl-C (optional)
print("\nCtrl-C to close")
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break