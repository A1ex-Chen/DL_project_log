def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    """ Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    """
    indices = []
    distances = []
    for p1, p2 in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)
    return indices, distances
