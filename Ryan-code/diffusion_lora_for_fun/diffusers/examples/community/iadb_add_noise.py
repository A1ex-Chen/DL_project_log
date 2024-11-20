def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    alpha: torch.Tensor) ->torch.Tensor:
    return original_samples * alpha + noise * (1 - alpha)
