def get_estimate(self, absolute=False) ->np.ndarray:
    """Get the position estimate of the object from the Kalman filter in an absolute or relative format.

        Parameters
        ----------
        absolute : bool, optional
            If true the coordinates are returned in absolute format, by default False, by default False.

        Returns
        -------
        np.ndarray
            An array of shape (self.num_points, self.dim_points) containing the position estimate of the object on each axis.

        Raises
        ------
        ValueError
            Alert if the coordinates are requested in absolute format but the tracker has no coordinate transformation.
        """
    positions = self.filter.x.T.flatten()[:self.dim_z].reshape(-1, self.
        dim_points)
    if self.abs_to_rel is None:
        if not absolute:
            return positions
        else:
            raise ValueError(
                "You must provide 'coord_transformations' to the tracker to get absolute coordinates"
                )
    elif absolute:
        return positions
    else:
        return self.abs_to_rel(positions)
