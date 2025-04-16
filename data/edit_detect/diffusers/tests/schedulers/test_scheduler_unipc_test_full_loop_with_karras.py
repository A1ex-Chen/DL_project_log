def test_full_loop_with_karras(self):
    sample = self.full_loop(use_karras_sigmas=True)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 0.2898) < 0.001
