def get_initial_kalman_filter_states(self, box: np.ndarray) ->dict[str, np.
    ndarray]:
    return {'x': np.array([box[0], box[1], box[2], box[3], 0, 0, 0, 0]),
        'P': np.eye(8), 'F': self._initialize_state_transition_matrix(),
        'H': self._initialize_measurement_function(), 'R': self.
        _initialize_measurement_noise_covariance(), 'Q': self.
        _initialize_process_noise_covariance()}
