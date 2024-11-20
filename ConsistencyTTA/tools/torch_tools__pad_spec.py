def _pad_spec(fbank, target_length=1024):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]
    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]
    return fbank
