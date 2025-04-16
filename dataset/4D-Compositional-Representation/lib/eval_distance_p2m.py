def distance_p2m(points, mesh):
    """ Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist
