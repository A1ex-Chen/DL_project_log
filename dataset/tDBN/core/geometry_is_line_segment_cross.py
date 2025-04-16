def is_line_segment_cross(lines1, lines2):
    A = lines1[:, 0, :][:, np.newaxis, :]
    B = lines1[:, 1, :][:, np.newaxis, :]
    C = lines2[:, 0, :][np.newaxis, :, :]
    D = lines2[:, 1, :][np.newaxis, :, :]
    return np.logical_and(_ccw(A, C, D) != _ccw(B, C, D), _ccw(A, B, C) !=
        _ccw(A, B, D))
