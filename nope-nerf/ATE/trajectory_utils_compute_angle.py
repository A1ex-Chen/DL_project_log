def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2))
        ) * 180.0 / np.pi
