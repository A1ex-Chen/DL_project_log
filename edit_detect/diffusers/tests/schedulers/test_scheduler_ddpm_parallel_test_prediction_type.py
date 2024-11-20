def test_prediction_type(self):
    for prediction_type in ['epsilon', 'sample', 'v_prediction']:
        self.check_over_configs(prediction_type=prediction_type)
