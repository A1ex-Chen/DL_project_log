def test_custom_timesteps(self):
    for prediction_type in ['epsilon', 'sample', 'v_prediction']:
        for interpolation_type in ['linear', 'log_linear']:
            for final_sigmas_type in ['sigma_min', 'zero']:
                sample = self.full_loop(prediction_type=prediction_type,
                    interpolation_type=interpolation_type,
                    final_sigmas_type=final_sigmas_type)
                sample_custom_timesteps = self.full_loop_custom_timesteps(
                    prediction_type=prediction_type, interpolation_type=
                    interpolation_type, final_sigmas_type=final_sigmas_type)
                assert torch.sum(torch.abs(sample - sample_custom_timesteps)
                    ) < 1e-05, f'Scheduler outputs are not identical for prediction_type: {prediction_type}, interpolation_type: {interpolation_type} and final_sigmas_type: {final_sigmas_type}'
