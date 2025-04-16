def _approximate_quad(self, hull):
    """Approximate a convex hull to a quadrilateral by considering most distant points."""
    first_point = self._get_upper_left_corner(hull)
    second_point = self._farthest_point_from(first_point, hull)
    max_distance = 0
    third_point = None
    for pt in np.array(hull, dtype=np.float32):
        dist = cv2.pointPolygonTest(np.array([first_point, second_point],
            dtype=np.float32), pt[0], True)
        if abs(dist) > max_distance:
            max_distance = abs(dist)
            third_point = pt[0]
    fourth_point = self._farthest_point_from(third_point, hull)
    quadrilateral = np.array([first_point, second_point, third_point,
        fourth_point])
    return quadrilateral
