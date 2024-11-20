def test_solver_order_and_type(self):
    for solver_type in ['bh1', 'bh2']:
        for order in [1, 2, 3]:
            for prediction_type in ['epsilon', 'sample']:
                self.check_over_configs(solver_order=order, solver_type=
                    solver_type, prediction_type=prediction_type)
                sample = self.full_loop(solver_order=order, solver_type=
                    solver_type, prediction_type=prediction_type)
                assert not torch.isnan(sample).any(
                    ), 'Samples have nan numbers'
