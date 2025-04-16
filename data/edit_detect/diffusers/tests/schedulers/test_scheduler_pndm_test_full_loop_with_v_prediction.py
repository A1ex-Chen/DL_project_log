def test_full_loop_with_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction')
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 67.3986) < 0.01
    assert abs(result_mean.item() - 0.0878) < 0.001
