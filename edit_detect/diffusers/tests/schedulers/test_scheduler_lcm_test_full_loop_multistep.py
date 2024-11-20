def test_full_loop_multistep(self):
    sample = self.full_loop(num_inference_steps=10)
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 197.7616) < 0.001
    assert abs(result_mean.item() - 0.2575) < 0.001
