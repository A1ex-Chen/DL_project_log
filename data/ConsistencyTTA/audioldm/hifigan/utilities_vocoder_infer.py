def vocoder_infer(mels, vocoder, allow_grad=False, lengths=None):
    vocoder.eval()
    if allow_grad:
        wavs = vocoder(mels).squeeze(1).float()
        wavs = wavs - (wavs.max() + wavs.min()) / 2
    else:
        with torch.no_grad():
            wavs = vocoder(mels).squeeze(1).float()
            wavs = wavs - (wavs.max() + wavs.min()) / 2
            wavs = (wavs.cpu().numpy() * 32768).astype('int16')
    if lengths is not None:
        wavs = wavs[:, :lengths]
    return wavs
