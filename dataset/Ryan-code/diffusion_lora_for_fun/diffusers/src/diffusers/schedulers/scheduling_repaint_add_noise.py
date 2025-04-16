def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.IntTensor) ->torch.Tensor:
    raise NotImplementedError(
        'Use `DDPMScheduler.add_noise()` to train for sampling with RePaint.')
