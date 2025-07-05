import trimesh
import numpy as np

mesh = trimesh.load(r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\STL_file\longwedge.stl")

print("Centroid:", mesh.centroid)

# direction_data_list = np.load(r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\pvtrace-master\detected_ray_directions_pyramid.npy")