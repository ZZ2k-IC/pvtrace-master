import time
import sys
import functools
import numpy as np
from pvtrace import *

world = Node(
    name="world (air)",
    geometry=Sphere(
        radius=10.0,
        material=Material(refractive_index=1.0),
    ),
)

# Create a rectangular beam source
def constant_direction():
    return (0.0, 0.0, 1.0)
light = Node(
    name="Rectangular Light Beam",
    light=Light(
        position=functools.partial(rectangular_mask, 2.0, 1.0),  # 2x1 rectangle
        direction=constant_direction  # Narrow beam (30 degrees)
    ),
    parent=world,
)
light.location = (0, 0, -3)  # Position light below

# Create a planar detector to catch the beam
detector = create_planar_detector_node(
    name="Beam Detector",
    length=4.0,
    width=2.0,
    normal=(0, 0, 1),  # Normal pointing up
    detection_direction=(0, 0, 1),  # Detect rays coming from below (upward)
    angle_tolerance=np.pi/2,  # Accept rays within 60 degrees
    parent=world
)
detector.translate((0, 0, 3))  # Position detector above

# Change zmq_url here to be the address of your meshcat-server!
renderer = MeshcatRenderer(
    zmq_url="tcp://127.0.0.1:6000", wireframe=True, open_browser=True
)
scene = Scene(world)
renderer.render(scene)

print("Starting simulation...")
initial_count = detector.detector_delegate.detected_count
for ray in scene.emit(100):
    steps = photon_tracer.follow(scene, ray)
    path, events = zip(*steps)
    
    # Check if this ray was detected
    if detector.detector_delegate.detected_count > initial_count:
        # Ray was detected, remove the reflection segment
        path = path[:-1]
        initial_count = detector.detector_delegate.detected_count
    
    renderer.add_ray_path(path)
    time.sleep(0.05)
    
# Print detection results
print(f"\nDetection Results:")
print(f"Total rays detected: {detector.detector_delegate.detected_count}")
print(f"Detection rate: {detector.detector_delegate.detected_count/100*100:.1f}%")

# Show some detected ray information
if detector.detector_delegate.detected_rays:
    print(f"\nFirst few detected rays:")
    for i, ray_info in enumerate(detector.detector_delegate.detected_rays[:5]):
        pos = ray_info['position']
        direction = ray_info['direction']
        print(f"Ray {i+1}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
              f"dir=({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")

# Wait for Ctrl-C to terminate the script; keep the window open
print("\nCtrl-C to close")
while True:
    try:
        time.sleep(0.3)
    except KeyboardInterrupt:
        sys.exit()