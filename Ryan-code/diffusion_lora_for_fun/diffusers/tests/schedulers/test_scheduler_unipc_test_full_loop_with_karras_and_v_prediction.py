def test_full_loop_with_karras_and_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction',
        use_karras_sigmas=True)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.1944) < 0.001
