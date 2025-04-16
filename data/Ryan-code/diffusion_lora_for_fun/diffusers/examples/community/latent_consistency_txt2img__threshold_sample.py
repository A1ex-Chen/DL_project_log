def _threshold_sample(self, sample: torch.Tensor) ->torch.Tensor:
    """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        https://arxiv.org/abs/2205.11487
        """
    dtype = sample.dtype
    batch_size, channels, height, width = sample.shape
    if dtype not in (torch.float32, torch.float64):
        sample = sample.float()
    sample = sample.reshape(batch_size, channels * height * width)
    abs_sample = sample.abs()
    s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio,
        dim=1)
    s = torch.clamp(s, min=1, max=self.config.sample_max_value)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s
    sample = sample.reshape(batch_size, channels, height, width)
    sample = sample.to(dtype)
    return sample
