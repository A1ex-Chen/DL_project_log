def test_full_loop_multistep_deter(self):
    sample = self.full_loop(num_inference_steps=10)
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 181.204) < 0.001
    assert abs(result_mean.item() - 0.2359) < 0.001
