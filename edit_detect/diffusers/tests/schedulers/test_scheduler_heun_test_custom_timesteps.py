def test_custom_timesteps(self):
    for prediction_type in ['epsilon', 'sample', 'v_prediction']:
        for timestep_spacing in ['linspace', 'leading']:
            sample = self.full_loop(prediction_type=prediction_type,
                timestep_spacing=timestep_spacing)
            sample_custom_timesteps = self.full_loop_custom_timesteps(
                prediction_type=prediction_type, timestep_spacing=
                timestep_spacing)
            assert torch.sum(torch.abs(sample - sample_custom_timesteps)
                ) < 1e-05, f'Scheduler outputs are not identical for prediction_type: {prediction_type}, timestep_spacing: {timestep_spacing}'
