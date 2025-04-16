def test_full_loop_with_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction')
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 1394.2185) < 0.01
    assert abs(result_mean.item() - 1.8154) < 0.001
