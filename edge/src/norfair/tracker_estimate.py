@property
def estimate(self) ->np.ndarray:
    """Get the position estimate of the object from the Kalman filter.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the position estimate of the object on each axis.
        """
    return self.get_estimate()
