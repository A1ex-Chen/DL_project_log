def stft(self, x):
    """
        Perform STFT and convert to magnitude spectrogram.
        Adapted from https://github.com/facebookresearch/denoiser/blob/
        8afd7c166699bb3c8b2d95b6dd706f71e1075df0/denoiser/stft_loss.py#L17

        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        Returns:
            Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
        """
    x_stft = torch.view_as_real(torch.stft(x.double(), self.fft_size, self.
        shift_size, self.win_length, self.window, return_complex=True))
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    mag = real ** 2 + imag ** 2
    return torch.clamp(mag, min=1e-08).sqrt().transpose(2, 1).float()
