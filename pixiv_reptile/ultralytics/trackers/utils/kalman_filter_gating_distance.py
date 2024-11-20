def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
    measurements: np.ndarray, only_position: bool=False, metric: str='maha'
    ) ->np.ndarray:
    """
        Compute gating distance between state distribution and measurements. A suitable distance threshold can be
        obtained from `chi2inv95`. If `only_position` is False, the chi-square distribution has 4 degrees of freedom,
        otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8 dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8 dimensional).
            measurements (ndarray): An Nx4 matrix of N measurements, each in format (x, y, a, h) where (x, y)
                is the bounding box center position, a the aspect ratio, and h the height.
            only_position (bool, optional): If True, distance computation is done with respect to the bounding box
                center position only. Defaults to False.
            metric (str, optional): The metric to use for calculating the distance. Options are 'gaussian' for the
                squared Euclidean distance and 'maha' for the squared Mahalanobis distance. Defaults to 'maha'.

        Returns:
            (np.ndarray): Returns an array of length N, where the i-th element contains the squared distance between
                (mean, covariance) and `measurements[i]`.
        """
    mean, covariance = self.project(mean, covariance)
    if only_position:
        mean, covariance = mean[:2], covariance[:2, :2]
        measurements = measurements[:, :2]
    d = measurements - mean
    if metric == 'gaussian':
        return np.sum(d * d, axis=1)
    elif metric == 'maha':
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True,
            check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)
    else:
        raise ValueError('Invalid distance metric')
