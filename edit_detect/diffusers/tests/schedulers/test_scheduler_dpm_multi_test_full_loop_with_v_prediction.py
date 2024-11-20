def test_full_loop_with_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction')
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.2251) < 0.001
