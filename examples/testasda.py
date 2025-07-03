import trimesh

mesh = trimesh.load(r"C:\Users\Zedd\Downloads\HORN.stl")

print("Centroid:", mesh.centroid)