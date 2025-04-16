def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = torch.squeeze(log_magnitudes_stft, 0).numpy().astype(
        np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy
