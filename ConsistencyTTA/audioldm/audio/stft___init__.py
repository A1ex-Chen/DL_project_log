def __init__(self, filter_length, hop_length, win_length, n_mel_channels,
    sampling_rate, mel_fmin, mel_fmax):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=filter_length,
        n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)
