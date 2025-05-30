def test_full_loop_no_noise(self):
    sample = self.full_loop()
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 198.1318) < 0.01
    assert abs(result_mean.item() - 0.258) < 0.001
