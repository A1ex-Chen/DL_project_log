def get_best_yaw(C):
    """
    maximize trace(Rz(theta) * C)
    """
    assert C.shape == (3, 3)
    A = C[0, 1] - C[1, 0]
    B = C[0, 0] + C[1, 1]
    theta = np.pi / 2 - np.arctan2(B, A)
    return theta
