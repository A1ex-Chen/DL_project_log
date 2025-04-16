def test_full_loop_with_v_prediction(self):
    sample = self.full_loop(prediction_type='v_prediction')
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['cpu', 'mps']:
        assert abs(result_sum.item() - 4.6934e-07) < 0.01
        assert abs(result_mean.item() - 6.1112e-10) < 0.001
    else:
        assert abs(result_sum.item() - 4.693428650170972e-07) < 0.01
        assert abs(result_mean.item() - 0.0002) < 0.001
