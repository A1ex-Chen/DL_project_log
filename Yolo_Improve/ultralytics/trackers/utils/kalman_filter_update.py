def update(self, mean, covariance, measurement) ->tuple:
    """
        Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8 dimensional).
            measurement (ndarray): The 4 dimensional measurement vector (x, y, w, h), where (x, y) is the center
                position, w the width, and h the height of the bounding box.

        Returns:
            (tuple[ndarray, ndarray]): Returns the measurement-corrected state distribution.
        """
    return super().update(mean, covariance, measurement)
