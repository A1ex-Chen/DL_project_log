def rotate_point_cloud(points, rotation_matrix=None):
    """Input: (n,3), Output: (n,3)"""
    if rotation_matrix is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
        sinval, cosval = np.sin(rotation_angle), np.cos(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0], [-sinval, cosval, 
            0], [0, 0, 1]])
    ctr = points.mean(axis=0)
    rotated_data = np.dot(points - ctr, rotation_matrix) + ctr
    return rotated_data, rotation_matrix
