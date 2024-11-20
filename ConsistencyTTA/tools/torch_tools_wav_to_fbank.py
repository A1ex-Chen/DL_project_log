def wav_to_fbank(waveforms, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveforms, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)
    fbank = _pad_spec(fbank, target_length)
    log_magnitudes_stft = _pad_spec(log_magnitudes_stft, target_length)
    return fbank, log_magnitudes_stft
