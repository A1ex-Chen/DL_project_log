def _threshold_sample(self, sample: torch.FloatTensor) ->torch.FloatTensor:
    dynamic_max_val = sample.flatten(1).abs().quantile(self.config.
        dynamic_thresholding_ratio, dim=1).clamp_min(self.config.
        sample_max_value).view(-1, *([1] * (sample.ndim - 1)))
    return sample.clamp(-dynamic_max_val, dynamic_max_val) / dynamic_max_val
