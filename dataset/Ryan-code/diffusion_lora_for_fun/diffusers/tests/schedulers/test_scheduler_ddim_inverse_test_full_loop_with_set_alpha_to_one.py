def test_full_loop_with_set_alpha_to_one(self):
    sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 539.9622) < 0.01
    assert abs(result_mean.item() - 0.7031) < 0.001
