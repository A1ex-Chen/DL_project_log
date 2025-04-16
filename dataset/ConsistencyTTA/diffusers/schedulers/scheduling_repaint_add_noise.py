def add_noise(self, original_samples: torch.FloatTensor, noise: torch.
    FloatTensor, timesteps: torch.IntTensor) ->torch.FloatTensor:
    raise NotImplementedError(
        'Use `DDPMScheduler.add_noise()` to train for sampling with RePaint.')
