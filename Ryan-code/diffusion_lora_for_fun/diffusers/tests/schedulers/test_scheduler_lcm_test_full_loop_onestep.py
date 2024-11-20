def test_full_loop_onestep(self):
    sample = self.full_loop(num_inference_steps=1)
    result_sum = torch.sum(torch.abs(sample))
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_sum.item() - 18.7097) < 0.001
    assert abs(result_mean.item() - 0.0244) < 0.001
