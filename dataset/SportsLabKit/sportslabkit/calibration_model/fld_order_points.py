def order_points(self, pts):
    """Order the points in clockwise order starting from top-left."""
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    ordered_pts = pts[np.argsort(angles)]
    return ordered_pts
