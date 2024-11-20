def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    waveform = read_wav_file(filename, target_length * 160)
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)
    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length)
    return fbank, log_magnitudes_stft, waveform
