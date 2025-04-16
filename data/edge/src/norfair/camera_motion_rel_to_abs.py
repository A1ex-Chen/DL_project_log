def rel_to_abs(self, points: np.ndarray):
    ones = np.ones((len(points), 1))
    points_with_ones = np.hstack((points, ones))
    points_transformed = points_with_ones @ self.inverse_homography_matrix.T
    points_transformed = points_transformed / points_transformed[:, -1
        ].reshape(-1, 1)
    return points_transformed[:, :2]
