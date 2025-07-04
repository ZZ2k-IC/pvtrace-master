import trimesh

mesh = trimesh.load(r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\STL_file\pyramid.stl")

print("Centroid:", mesh.centroid)