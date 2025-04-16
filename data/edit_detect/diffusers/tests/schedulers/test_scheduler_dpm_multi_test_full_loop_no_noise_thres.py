def test_full_loop_no_noise_thres(self):
    sample = self.full_loop(thresholding=True, dynamic_thresholding_ratio=
        0.87, sample_max_value=0.5)
    result_mean = torch.mean(torch.abs(sample))
    assert abs(result_mean.item() - 1.1364) < 0.001
