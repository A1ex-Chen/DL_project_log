def project(self, mean, covariance) ->tuple:
    """
        Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).

        Returns:
            (tuple[ndarray, ndarray]): Returns the projected mean and covariance matrix of the given state estimate.
        """
    std = [self._std_weight_position * mean[2], self._std_weight_position *
        mean[3], self._std_weight_position * mean[2], self.
        _std_weight_position * mean[3]]
    innovation_cov = np.diag(np.square(std))
    mean = np.dot(self._update_mat, mean)
    covariance = np.linalg.multi_dot((self._update_mat, covariance, self.
        _update_mat.T))
    return mean, covariance + innovation_cov
