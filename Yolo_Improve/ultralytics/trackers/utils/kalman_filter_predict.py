def predict(self, mean, covariance) ->tuple:
    """
        Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object state at the previous time step.
            covariance (ndarray): The 8x8 dimensional covariance matrix of the object state at the previous time step.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector and covariance matrix of the predicted state. Unobserved
                velocities are initialized to 0 mean.
        """
    std_pos = [self._std_weight_position * mean[2], self.
        _std_weight_position * mean[3], self._std_weight_position * mean[2],
        self._std_weight_position * mean[3]]
    std_vel = [self._std_weight_velocity * mean[2], self.
        _std_weight_velocity * mean[3], self._std_weight_velocity * mean[2],
        self._std_weight_velocity * mean[3]]
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    mean = np.dot(mean, self._motion_mat.T)
    covariance = np.linalg.multi_dot((self._motion_mat, covariance, self.
        _motion_mat.T)) + motion_cov
    return mean, covariance
