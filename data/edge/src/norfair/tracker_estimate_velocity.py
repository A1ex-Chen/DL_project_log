@property
def estimate_velocity(self) ->np.ndarray:
    """Get the velocity estimate of the object from the Kalman filter. This velocity is in the absolute coordinate system.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the velocity estimate of the object on each axis.
        """
    return self.filter.x.T.flatten()[self.dim_z:].reshape(-1, self.dim_points)
