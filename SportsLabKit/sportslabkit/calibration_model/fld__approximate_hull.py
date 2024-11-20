def _approximate_hull(self, contour):
    hull = cv2.convexHull(contour)
    return hull
