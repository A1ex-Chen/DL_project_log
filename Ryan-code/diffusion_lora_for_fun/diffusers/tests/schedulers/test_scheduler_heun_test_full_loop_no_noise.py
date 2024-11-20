def test_full_loop_no_noise(self):
    sample = self.full_loop()
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    if torch_device in ['cpu', 'mps']:
        assert abs(result_sum.item() - 0.1233) < 0.01
        assert abs(result_mean.item() - 0.0002) < 0.001
    else:
        assert abs(result_sum.item() - 0.1233) < 0.01
        assert abs(result_mean.item() - 0.0002) < 0.001
