def test_full_loop_with_no_set_alpha_to_one(self):
    sample = self.full_loop(set_alpha_to_one=False, beta_start=0.01)
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 186.9482) < 0.01
    assert abs(result_mean.item() - 0.2434) < 0.001