def _arrange_points_clockwise(self, points):
    """Arrange the given points in clockwise order starting from top-left."""
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered_points = points[np.argsort(angles)]
    return ordered_points
