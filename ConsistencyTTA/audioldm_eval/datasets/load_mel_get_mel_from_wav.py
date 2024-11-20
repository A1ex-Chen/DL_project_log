def get_mel_from_wav(self, audio):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, energy = self._stft.mel_spectrogram(audio, normalize_fun=torch
        .log10)
    melspec = melspec * 20 - 20
    melspec = (melspec + 100) / 100
    melspec = torch.clip(melspec, min=0, max=1.0)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, energy
