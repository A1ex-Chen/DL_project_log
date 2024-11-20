def test_custom_timesteps(self):
    for prediction_type in ['epsilon', 'sample', 'v_prediction']:
        for lower_order_final in [True, False]:
            for final_sigmas_type in ['sigma_min', 'zero']:
                sample = self.full_loop(prediction_type=prediction_type,
                    lower_order_final=lower_order_final, final_sigmas_type=
                    final_sigmas_type)
                sample_custom_timesteps = self.full_loop_custom_timesteps(
                    prediction_type=prediction_type, lower_order_final=
                    lower_order_final, final_sigmas_type=final_sigmas_type)
                assert torch.sum(torch.abs(sample - sample_custom_timesteps)
                    ) < 1e-05, f'Scheduler outputs are not identical for prediction_type: {prediction_type}, lower_order_final: {lower_order_final} and final_sigmas_type: {final_sigmas_type}'
