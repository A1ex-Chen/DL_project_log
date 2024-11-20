def _farthest_point_from(self, reference_point, point_list):
    """Find the point in 'point_list' that is farthest from 'reference_point'."""
    max_dist = 0
    farthest_point = None
    for point in point_list:
        dist = cv2.norm(reference_point - point[0])
        if dist > max_dist:
            max_dist = dist
            farthest_point = point[0]
    return farthest_point
