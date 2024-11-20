def mel_spectrogram(self, y, normalize_fun=torch.log):
    """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
    assert torch.min(y.data) >= -1
    assert torch.max(y.data) <= 1
    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.detach()
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output, normalize_fun)
    energy = torch.norm(magnitudes, dim=1)
    return mel_output, energy
