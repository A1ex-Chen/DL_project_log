def test_prediction_type(self):
    for prediction_type in ['epsilon', 'v_prediction']:
        self.check_over_configs(time_step=self.default_valid_timestep,
            prediction_type=prediction_type)
