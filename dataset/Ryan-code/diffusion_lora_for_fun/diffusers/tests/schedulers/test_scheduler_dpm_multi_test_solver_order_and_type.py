def test_solver_order_and_type(self):
    for algorithm_type in ['dpmsolver', 'dpmsolver++', 'sde-dpmsolver',
        'sde-dpmsolver++']:
        for solver_type in ['midpoint', 'heun']:
            for order in [1, 2, 3]:
                for prediction_type in ['epsilon', 'sample']:
                    if algorithm_type in ['sde-dpmsolver', 'sde-dpmsolver++']:
                        if order == 3:
                            continue
                    else:
                        self.check_over_configs(solver_order=order,
                            solver_type=solver_type, prediction_type=
                            prediction_type, algorithm_type=algorithm_type)
                    sample = self.full_loop(solver_order=order, solver_type
                        =solver_type, prediction_type=prediction_type,
                        algorithm_type=algorithm_type)
                    assert not torch.isnan(sample).any(
                        ), 'Samples have nan numbers'
