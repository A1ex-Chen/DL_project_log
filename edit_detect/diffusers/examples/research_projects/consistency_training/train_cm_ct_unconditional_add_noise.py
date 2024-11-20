def add_noise(original_samples: torch.Tensor, noise: torch.Tensor,
    timesteps: torch.Tensor):
    sigmas = timesteps.to(device=original_samples.device, dtype=
        original_samples.dtype)
    while len(sigmas.shape) < len(original_samples.shape):
        sigmas = sigmas.unsqueeze(-1)
    noisy_samples = original_samples + noise * sigmas
    return noisy_samples
