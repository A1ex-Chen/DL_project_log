def test_custom_timesteps(self):
    for algorithm_type in ['dpmsolver++', 'sde-dpmsolver++']:
        for prediction_type in ['epsilon', 'sample', 'v_prediction']:
            for final_sigmas_type in ['sigma_min', 'zero']:
                sample = self.full_loop(algorithm_type=algorithm_type,
                    prediction_type=prediction_type, final_sigmas_type=
                    final_sigmas_type)
                sample_custom_timesteps = self.full_loop_custom_timesteps(
                    algorithm_type=algorithm_type, prediction_type=
                    prediction_type, final_sigmas_type=final_sigmas_type)
                assert torch.sum(torch.abs(sample - sample_custom_timesteps)
                    ) < 1e-05, f'Scheduler outputs are not identical for algorithm_type: {algorithm_type}, prediction_type: {prediction_type} and final_sigmas_type: {final_sigmas_type}'
