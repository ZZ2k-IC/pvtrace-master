from pvtrace import *
import time
import functools
import numpy as np

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

box = Node(
    name="Santovac 5",
    geometry=Box(
        (1, 1, 0.2),
        material=Material(refractive_index=1.63),
    ),
    parent=world
)




cylinder = Node(
    name="Cylinder (fused quartz)",
    geometry=Cylinder(
        length=6,
        radius=0.5,
        material=Material(refractive_index=1.4585),
    ),
    parent=world
)
cylinder.translate((0, 0, 3.1))

# Add source of photons
light = Node(
    name="Light (555nm)",
    parent=world,
    light=Light(
        position=functools.partial(rectangular_mask, 0.32, 0.26),
        direction=functools.partial(lambertian, np.pi / 3) # Maximum beam angle is 60 degrees.
    )
)


# Add detector at bottom of cylinder (z=0.1) - detects rays coming from above
bottom_detector = create_planar_detector_node(
    name="Bottom Detector",
    length=1.0,  # Larger than cylinder radius (0.5) to catch all rays
    width=1.0,
    normal=(0, 0, 1),  # Normal pointing up
    detection_direction=(0, 0, -1),  # Detect rays coming from above (downward)
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

bottom_initial_count = bottom_detector.detector_delegate.detected_count
top_initial_count = top_detector.detector_delegate.detected_count


for ray in scene.emit(rays_num):  # Note: using 'viewer' not 'scene'
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

# Keep the script alive until Ctrl-C (optional)
print("\nCtrl-C to close")
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break