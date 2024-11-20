def get_angles(x0, com, orientation, dimensions):
    vector = orientationVector(x0, com, dimensions)
    dot_product = np.dot(vector, orientation)
    cross_product = np.cross(vector, orientation)
    angle = np.arctan2(cross_product, dot_product)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    return angle / (2 * np.pi)
