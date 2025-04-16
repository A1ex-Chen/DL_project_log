def test_thresholding(self):
    self.check_over_configs(thresholding=False)
    for order in [1, 2, 3]:
        for solver_type in ['bh1', 'bh2']:
            for threshold in [0.5, 1.0, 2.0]:
                for prediction_type in ['epsilon', 'sample']:
                    self.check_over_configs(thresholding=True,
                        prediction_type=prediction_type, sample_max_value=
                        threshold, solver_order=order, solver_type=solver_type)
