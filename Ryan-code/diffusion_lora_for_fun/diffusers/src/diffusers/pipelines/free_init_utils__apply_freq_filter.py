def _apply_freq_filter(self, x: torch.Tensor, noise: torch.Tensor,
    low_pass_filter: torch.Tensor) ->torch.Tensor:
    """Noise reinitialization."""
    x_freq = fft.fftn(x, dim=(-3, -2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-3, -2, -1))
    noise_freq = fft.fftn(noise, dim=(-3, -2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-3, -2, -1))
    high_pass_filter = 1 - low_pass_filter
    x_freq_low = x_freq * low_pass_filter
    noise_freq_high = noise_freq * high_pass_filter
    x_freq_mixed = x_freq_low + noise_freq_high
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real
    return x_mixed
