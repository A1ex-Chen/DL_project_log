def multi_predict(self, mean, covariance) ->tuple:
    """
        Run Kalman filter prediction step (Vectorized version).

        Args:
            mean (ndarray): The Nx8 dimensional mean matrix of the object states at the previous time step.
            covariance (ndarray): The Nx8x8 covariance matrix of the object states at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
    std_pos = [self._std_weight_position * mean[:, 2], self.
        _std_weight_position * mean[:, 3], self._std_weight_position * mean
        [:, 2], self._std_weight_position * mean[:, 3]]
    std_vel = [self._std_weight_velocity * mean[:, 2], self.
        _std_weight_velocity * mean[:, 3], self._std_weight_velocity * mean
        [:, 2], self._std_weight_velocity * mean[:, 3]]
    sqr = np.square(np.r_[std_pos, std_vel]).T
    motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
    motion_cov = np.asarray(motion_cov)
    mean = np.dot(mean, self._motion_mat.T)
    left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
    covariance = np.dot(left, self._motion_mat.T) + motion_cov
    return mean, covariance
