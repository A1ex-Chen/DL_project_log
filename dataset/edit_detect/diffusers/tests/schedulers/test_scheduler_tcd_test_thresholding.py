def test_thresholding(self):
    self.check_over_configs(time_step=self.default_valid_timestep,
        thresholding=False)
    for threshold in [0.5, 1.0, 2.0]:
        for prediction_type in ['epsilon', 'v_prediction']:
            self.check_over_configs(time_step=self.default_valid_timestep,
                thresholding=True, prediction_type=prediction_type,
                sample_max_value=threshold)
