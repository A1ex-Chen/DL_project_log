def _initialize_process_noise_covariance(self) ->np.ndarray:
    q = np.array([[self.dt ** 4 / 4, 0, 0, 0, self.dt ** 3 / 2, 0, 0, 0], [
        0, self.dt ** 4 / 4, 0, 0, 0, self.dt ** 3 / 2, 0, 0], [0, 0, self.
        dt ** 4 / 4, 0, 0, 0, self.dt ** 3 / 2, 0], [0, 0, 0, self.dt ** 4 /
        4, 0, 0, 0, self.dt ** 3 / 2], [self.dt ** 3 / 2, 0, 0, 0, self.dt **
        2, 0, 0, 0], [0, self.dt ** 3 / 2, 0, 0, 0, self.dt ** 2, 0, 0], [0,
        0, self.dt ** 3 / 2, 0, 0, 0, self.dt ** 2, 0], [0, 0, 0, self.dt **
        3 / 2, 0, 0, 0, self.dt ** 2]])
    return q * self.process_noise
