def _initialize_state_transition_matrix(self) ->np.ndarray:
    return np.array([[1, 0, 0, 0, self.dt, 0, 0, 0], [0, 1, 0, 0, 0, self.
        dt, 0, 0], [0, 0, 1, 0, 0, 0, self.dt, 0], [0, 0, 0, 1, 0, 0, 0,
        self.dt], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0,
        0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
