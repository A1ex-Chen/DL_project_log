def check_mesh_contains(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    return contains
