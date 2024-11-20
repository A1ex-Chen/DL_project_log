def __call__(self, curr_pts: np.ndarray, prev_pts: np.ndarray) ->Tuple[bool,
    HomographyTransformation]:
    homography_matrix, points_used = cv2.findHomography(prev_pts, curr_pts,
        method=self.method, ransacReprojThreshold=self.
        ransac_reproj_threshold, maxIters=self.max_iters, confidence=self.
        confidence)
    proportion_points_used = np.sum(points_used) / len(points_used)
    update_prvs = (proportion_points_used < self.
        proportion_points_used_threshold)
    try:
        homography_matrix = homography_matrix @ self.data
    except (TypeError, ValueError):
        pass
    if update_prvs:
        self.data = homography_matrix
    return update_prvs, HomographyTransformation(homography_matrix)
