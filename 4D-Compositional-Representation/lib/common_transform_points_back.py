def transform_points_back(points, transform):
    """ Inverts the transformation.

    Args:
        points (tensor): points tensor
        transform (tensor): transformation matrices
    """
    assert points.size(2) == 3
    assert transform.size(1) == 3
    assert points.size(0) == transform.size(0)
    if transform.size(2) == 4:
        R = transform[:, :, :3]
        t = transform[:, :, 3:]
        points_out = points - t.transpose(1, 2)
        points_out = points_out @ b_inv(R.transpose(1, 2))
    elif transform.size(2) == 3:
        K = transform
        points_out = points @ b_inv(K.transpose(1, 2))
    return points_out
