from pvtrace import *
import time
import functools
import numpy as np

# Add nodes to the scene graph
world = Node(
    name="World",
    geometry=Sphere(
        radius=15.0,
        material=Material(refractive_index=1.0),
    )
)

box1 = Node(
    name="box1",
    geometry=Plane(
        (1, 1),
        material=Material(refractive_index=1.6),
    ),
    parent=world
)
box1.translate((8, 8, 0))

box2 = Node(
    name="box2",
    geometry=Plane(
        (1, 1),
        material=Material(refractive_index=1.6),
    ),
    parent=world
)
box2.translate((-8, -8, 0))

cylinder = Node(
    name="A",
    geometry=Cylinder(
        length=10,
        radius=0.5,
        material=Material(refractive_index=1.5),
    ),
    parent=world
)
cylinder.translate((0, 0, 5.1))

# Add source of photons
light = Node(
    name="Light (555nm)",
    parent=world,
    light=Light(position=functools.partial(rectangular_mask, 0.32, 0.26),
        direction=functools.partial(
            lambertian, np.pi / 3
        )
    )
)


# Use meshcat to render the scene (optional)
viewer = MeshcatRenderer(open_browser=True, transparency=False, opacity=0.5, wireframe=True)
scene = Scene(world)
viewer.render(scene)

print("Starting simulation...")
start_t = time.time()  # Start timing here

for ray in scene.emit(1000):
    history = photon_tracer.follow(scene, ray)
    path, events = zip(*history)
    viewer.add_ray_path(path)  

# Print timing results
print(f"Took {time.time() - start_t}s.")

# Keep the script alive until Ctrl-C (optional)
print("\nCtrl-C to close")
while True:
    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        break