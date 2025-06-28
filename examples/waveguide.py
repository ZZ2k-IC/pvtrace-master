from pvtrace import *
import numpy as np
import trimesh
import functools
import pvtrace.algorithm.photon_tracer as photon_tracer
from pvtrace.algorithm.photon_tracer import next_hit, Event, find_container

original_next_hit = next_hit
original_follow = photon_tracer.follow
# MINIMAL FIX: Just exclude detectors from container detection
original_find_container = find_container

def corrected_find_container(intersections):
    """Fixed container detection that ignores detectors"""
    if len(intersections) == 0:
        return None
    if len(intersections) == 1:
        return intersections[0].hit
    
    # Filter out detectors - they can't be containers
    container_candidates = []
    for intersection in intersections:
        if "Detector" not in intersection.hit.name:
            container_candidates.append(intersection)
    
    # If no valid containers found, use World
    if len(container_candidates) == 0:
        for intersection in intersections:
            if intersection.hit.name == "World":
                return intersection.hit
        return intersections[0].hit  # Fallback
    
    # Use original logic on filtered candidates
    if len(container_candidates) == 1:
        return container_candidates[0].hit
    return original_find_container(container_candidates)

# Apply the fix
photon_tracer.find_container = corrected_find_container

def priority_next_hit(scene, ray):
    """Modified next_hit that prioritizes waveguide over absorber when overlapping"""
    
    result = original_next_hit(scene, ray)
    if result is None:
        return None
        
    hit, (container, adjacent), point, full_distance = result
    
    # PRIORITY RULE: If ray is inside waveguide, ignore absorber intersections
    if container.name == "Waveguide":
        # Ray is inside waveguide - filter out absorber intersections
        intersections = scene.intersections(ray.position, ray.direction)
        intersections = [x for x in intersections if not np.isclose(x.distance, 0.0)]
        intersections = [x.to(scene.root) for x in intersections]
        
        # Remove absorber intersections when inside waveguide
        filtered_intersections = []
        for intersection in intersections:
            if "Absorber" not in intersection.hit.name:
                filtered_intersections.append(intersection)
        
        if filtered_intersections:
            # Sort by distance and take closest non-absorber intersection
            filtered_intersections.sort(key=lambda x: x.distance)
            hit = filtered_intersections[0].hit
            point = filtered_intersections[0].point
            full_distance = filtered_intersections[0].distance
            
            # Recalculate adjacent for the new hit
            if hit == container:
                # Ray hitting waveguide surface from inside
                adjacent = scene.root  # World
            else:
                # Ray hitting something else
                adjacent = hit
                
            return hit, (container, adjacent), point, full_distance
    
    # For all other cases, use original result
    return result

photon_tracer.next_hit = priority_next_hit


def corrected_follow(scene, ray, maxsteps=1000, maxpathlength=np.inf, emit_method='kT'):
    count = 0
    history = [(ray, (None,None,None), Event.GENERATE)]
    
    while True:
        count += 1
        if count > maxsteps or ray.travelled > maxpathlength:
            history.append((ray, (None,None,None), Event.KILL))
            break

        if ray.direction == (0.0, 0.0, 0.0):
            history.append((ray, (None,None,None), Event.DETECT))
            break
    
        info = next_hit(scene, ray)
        if info is None:
            history.append((ray, (None,None,None), Event.EXIT))
            break

        hit, (container, adjacent), point, full_distance = info
        if hit is scene.root:
            history.append((ray.propagate(full_distance), (None,None,None), Event.EXIT))
            break

        # FIX: Correct adjacent detection for waveguide surfaces
        if hit.name == "Waveguide" and container.name == "Waveguide":
            # Ray is inside waveguide hitting waveguide surface
            # Adjacent should always be World (air), not absorber
            corrected_adjacent = world  # Use the world node
        else:
            # Use original adjacent for other cases
            corrected_adjacent = adjacent
        
        material = container.geometry.material
        absorbed, at_distance = material.is_absorbed(ray, full_distance)
        
        if absorbed and at_distance < full_distance:
            ray = ray.propagate(at_distance)
            component = material.component(ray.wavelength)
            if component is not None and component.is_radiative(ray):
                ray = component.emit(ray.representation(scene.root, container), method=emit_method)
                ray = ray.representation(container, scene.root)
                if isinstance(component, Luminophore):
                    event = Event.EMIT
                elif isinstance(component, Scatterer):
                    event = Event.SCATTER
                else:
                    event = Event.SCATTER
                history.append((ray, (None,None,None), event))
                continue
            else:
                history.append((ray, (None,None,None), Event.ABSORB))
                break
        else:
            ray = ray.propagate(full_distance)
            surface = hit.geometry.material.surface
            ray = ray.representation(scene.root, hit)
            
            # Use corrected adjacent for surface interactions
            if surface.is_reflected(ray, hit.geometry, container, corrected_adjacent):
                ray = surface.reflect(ray, hit.geometry, container, corrected_adjacent)
                ray = ray.representation(hit, scene.root)
                
                try:
                    local_pos = list(np.array(ray.position) - np.array(hit.location))
                    normal = hit.geometry.normal(local_pos)
                except:
                    normal = (None, None, None)
                    
                history.append((ray, normal, Event.REFLECT))
                continue
            else:
                ref_ray = surface.transmit(ray, hit.geometry, container, corrected_adjacent)
                if ref_ray is None:
                    history.append((ray, (None,None,None), Event.KILL))
                    break
                    
                ray = ref_ray
                ray = ray.representation(hit, scene.root)
                
                try:
                    local_pos = list(np.array(ray.position) - np.array(hit.location))
                    normal = hit.geometry.normal(local_pos)
                except:
                    normal = (None, None, None)
                    
                history.append((ray, normal, Event.TRANSMIT))
                continue
                
    return history

photon_tracer.follow = corrected_follow

# 1. Create the world node (always needed)
world = Node(
    name="World",
    geometry=Sphere(
        radius=20,
        material=Material(refractive_index=1.0)
        
    )
)

# 2. Create arbitrary absorber (example: box, cylinder, mesh)
absorber = Node(
    name="Absorber",
    geometry=Box(
        (3, 3, 1.2),
        material=Material(
            refractive_index=1.0,
            components=[Absorber(coefficient=100.0)]  # Set absorption here
        )
    ),
    parent=world
)
absorber.translate((0, 0, 7))

# 3. Create arbitrary waveguide (example: cylinder, mesh, etc)
waveguide = Node(
    name="Waveguide",
    geometry=Cylinder(
        length=6,
        radius=1.5,
        material=Material(
            refractive_index=1.5  # No absorber, just refract/reflect
        )
    ),
    parent=world
)
waveguide.translate((0, 0, 4))

# 4. Add a light source (customize position, direction, divergence, etc)
light = Node(
    name="Light",
    parent=world,
    light=Light(
        position=functools.partial(rectangular_mask, 1, 1),
        direction=functools.partial(lambertian, np.pi/3)
    )
)
light.translate((0, 0, 1.1))

# 5. Add detectors (planar, box, etc)
detector = create_planar_detector_node(
    name="Detector",
    length=3,
    width=3,
    normal=(0, 0, 1),
    detection_direction=(0, 0, -1),
    parent=world
)
detector.translate((0, 0, 6))



# 6. (Optional) Add mesh geometry
# mesh = trimesh.load("your_mesh.stl")
# mesh_node = Node(
#     name="MeshAbsorber",
#     geometry=Mesh(
#         trimesh=mesh,
#         material=Material(refractive_index=1.5, components=[Absorber(coefficient=5.0)])
#     ),
#     parent=world
# )

# 7. Run the simulation
viewer = MeshcatRenderer(open_browser=True, transparency=False, opacity=0.5, wireframe=True)
scene = Scene(world)
viewer.render(scene)


# Add this simple debug version:
rays_num = 100
absorb_count = 0
reflect_count = 0
transmit_count = 0
detect_count = 0

for ray in scene.emit(rays_num):
    steps = photon_tracer.follow(scene, ray)
    path, surfnorms, events = zip(*steps)
    
    for event in events:
        if event == Event.ABSORB:
            absorb_count += 1
        elif event == Event.REFLECT:
            reflect_count += 1
        elif event == Event.TRANSMIT:
            transmit_count += 1
        elif event == Event.DETECT:
            detect_count += 1
    
    viewer.add_ray_path(path)

print(f"ABSORB: {absorb_count}")
print(f"REFLECT: {reflect_count}")
print(f"TRANSMIT: {transmit_count}")
print(f"DETECT: {detect_count}")
print(f"Detector hits: {detector.detector_delegate.detected_count} / {rays_num}")