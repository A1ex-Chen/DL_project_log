def initiate(self, measurement: np.ndarray) ->tuple:
    """
        Create track from unassociated measurement.

        Args:
            measurement (ndarray): Bounding box coordinates (x, y, w, h) with center position (x, y), width, and height.

        Returns:
            (tuple[ndarray, ndarray]): Returns the mean vector (8 dimensional) and covariance matrix (8x8 dimensional)
                of the new track. Unobserved velocities are initialized to 0 mean.
        """
    mean_pos = measurement
    mean_vel = np.zeros_like(mean_pos)
    mean = np.r_[mean_pos, mean_vel]
    std = [2 * self._std_weight_position * measurement[2], 2 * self.
        _std_weight_position * measurement[3], 2 * self.
        _std_weight_position * measurement[2], 2 * self.
        _std_weight_position * measurement[3], 10 * self.
        _std_weight_velocity * measurement[2], 10 * self.
        _std_weight_velocity * measurement[3], 10 * self.
        _std_weight_velocity * measurement[2], 10 * self.
        _std_weight_velocity * measurement[3]]
    covariance = np.diag(np.square(std))
    return mean, covariance
