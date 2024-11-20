def test_prediction_type(self):
    for prediction_type in ['epsilon', 'v_prediction', 'sample']:
        self.check_over_configs(prediction_type=prediction_type)
