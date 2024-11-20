def _calculate_homography(self, src_points, dst_points):
    """Compute the transformation matrix between source and destination points."""
    ordered_src = self._arrange_points_clockwise(src_points)
    ordered_dst = self._arrange_points_clockwise(dst_points)
    H, _ = cv2.findHomography(ordered_src, ordered_dst, method=0)
    return H
