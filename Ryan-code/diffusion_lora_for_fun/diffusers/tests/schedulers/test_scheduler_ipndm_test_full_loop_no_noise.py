def test_full_loop_no_noise(self):
    sample = self.full_loop()
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 2540529) < 10