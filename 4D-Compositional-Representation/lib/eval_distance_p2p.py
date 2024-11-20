def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1,
            keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1,
            keepdims=True)
        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0],
            dtype=np.float32)
    return dist, normals_dot_product
