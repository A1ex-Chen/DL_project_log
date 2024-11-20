def chamfer_distance_naive(points1, points2):
    """ Naive implementation of the Chamfer distance.

    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set    
    """
    assert points1.size() == points2.size()
    batch_size, T, _ = points1.size()
    points1 = points1.view(batch_size, T, 1, 3)
    points2 = points2.view(batch_size, 1, T, 3)
    distances = (points1 - points2).pow(2).sum(-1)
    chamfer1 = distances.min(dim=1)[0].mean(dim=1)
    chamfer2 = distances.min(dim=2)[0].mean(dim=1)
    chamfer = chamfer1 + chamfer2
    return chamfer
