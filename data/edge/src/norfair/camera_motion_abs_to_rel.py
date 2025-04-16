def abs_to_rel(self, points: np.ndarray):
    ones = np.ones((len(points), 1))
    points_with_ones = np.hstack((points, ones))
    points_transformed = points_with_ones @ self.homography_matrix.T
    points_transformed = points_transformed / points_transformed[:, -1
        ].reshape(-1, 1)
    return points_transformed[:, :2]
