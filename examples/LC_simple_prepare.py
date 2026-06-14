from pvtrace import *
import time
import functools
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pvtrace.light.light import XZ_rectangular_mask
import os

# ============================================================================
# Generate a ray dataset inside a simple prism waveguide.
#
# Rays are launched from a rectangular source embedded in the waveguide and
# detected at the top exit surface. The detected ray positions, directions,
# and wavelengths are saved for later reuse via StoredRayLight.
#
# Length unit: cm
# ============================================================================

os.makedirs("stored_rays_info", exist_ok=True)

# Number of emitted rays
rays_num = 80000


# ============================================================================
# World
# ============================================================================

world = Node(
    name="World",
    geometry=Sphere(
        radius=100.0,
        material=Material(refractive_index=1.0),
    )
)


# ============================================================================
# Waveguide
# ============================================================================

LC = Node(
    name="Glass Prism Waveguide",
    geometry=Box(
        (5, 0.6, 73),
        material=Material(refractive_index=1.82),
    ),
    parent=world
)

# Align top surface close to z = 0
LC.location = (0, 0, -73 / 2 + 0.01)


# ============================================================================
# Perfect mirror helper
# ============================================================================

def addXYMirror(mirror_z, mirror_x, mirror_y, thickness=1e-3):
    """
    Create an ideal specular reflector parallel to the XY plane.
    """

    mirror = Node(
        name="xyMirror",
        geometry=Box(
            (mirror_x, mirror_y, thickness),
            material=Material(refractive_index=1.0)
        ),
        parent=world
    )

    mirror.location = (0, 0, mirror_z)

    # Disable absorption and scattering
    mirror.geometry.material.components = []

    class PerfectMirror(FresnelSurfaceDelegate):
        def reflectivity(self, surface, ray, geometry, container, adjacent):
            return 1.0

    mirror.geometry.material.surface = Surface(delegate=PerfectMirror())

    return mirror


# Mirror below the waveguide
bottom_mirror = addXYMirror(
    mirror_z=-73 - 0.01,
    mirror_x=5,
    mirror_y=0.6
)


# ============================================================================
# Light source
# ============================================================================

light_sideX = 4.9
light_sideZ = 40

light = Node(
    name="Light (555 nm)",
    parent=world,
    light=Light(
        position=functools.partial(
            XZ_rectangular_mask,
            light_sideX / 2,
            light_sideZ / 2
        ),
        direction=functools.partial(isotropic, np.pi)
    )
)

# Source centered inside waveguide
light.location = (0, 0, -73 / 2)


# ============================================================================
# Detector
# ============================================================================

top_detector = create_planar_detector_node(
    name="Top Detector",
    length=5,
    width=0.6,
    normal=(0, 0, 1),
    detection_direction=(0, 0, 1),
    parent=world
)

top_detector.translate((0, 0, 0))


# ============================================================================
# Scene visualisation
# ============================================================================

viewer = MeshcatRenderer(
    open_browser=True,
    transparency=False,
    opacity=0.5,
    wireframe=True
)

scene = Scene(world)
viewer.render(scene)


# ============================================================================
# Ray tracing
# ============================================================================

print("Starting simulation...")
start_t = time.time()

initial_ray_directions = []

for ray in scene.emit(rays_num):

    initial_ray_directions.append(ray.direction)

    steps = photon_tracer.follow(scene, ray)
    path, surface_info, events = zip(*steps)

    viewer.add_ray_path(path)

print(f"Took {time.time() - start_t:.2f} s.")


# ============================================================================
# Detection statistics
# ============================================================================

print("\nDetection Results:")
print(
    f"Top detector: "
    f"{top_detector.detector_delegate.detected_count} rays detected"
)

top_efficiency = (
    top_detector.detector_delegate.detected_count
    / rays_num
    * 100
)

print(f"Top detection efficiency: {top_efficiency:.1f}%")


# ============================================================================
# Export detected rays
# ============================================================================

all_detected_rays = top_detector.detector_delegate.detected_rays

print(all_detected_rays[0].keys())

detected_positions = np.array(
    [ray["position"] for ray in all_detected_rays]
)

detected_directions = np.array(
    [ray["direction"] for ray in all_detected_rays]
)

detected_wavelengths = np.array(
    [ray["wavelength"] for ray in all_detected_rays]
)

np.savez(
    "detected_rays.npz",
    positions=detected_positions,
    directions=detected_directions,
    wavelengths=detected_wavelengths
)

print(f"Saved {len(detected_positions)} detected rays")