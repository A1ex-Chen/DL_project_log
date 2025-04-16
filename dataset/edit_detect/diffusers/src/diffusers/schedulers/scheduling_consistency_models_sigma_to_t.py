def sigma_to_t(self, sigmas: Union[float, np.ndarray]):
    """
        Gets scaled timesteps from the Karras sigmas for input to the consistency model.

        Args:
            sigmas (`float` or `np.ndarray`):
                A single Karras sigma or an array of Karras sigmas.

        Returns:
            `float` or `np.ndarray`:
                A scaled input timestep or scaled input timestep array.
        """
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array(sigmas, dtype=np.float64)
    timesteps = 1000 * 0.25 * np.log(sigmas + 1e-44)
    return timesteps
