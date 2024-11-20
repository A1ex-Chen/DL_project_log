def stft(self, x):
    return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
        win_length=self.win_length, window=self.window.to(dtype=torch.float))
