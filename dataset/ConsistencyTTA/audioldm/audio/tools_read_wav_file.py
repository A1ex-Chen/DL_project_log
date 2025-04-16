def read_wav_file(filename, segment_length, target_sr=16000):
    waveform, orig_sr = sf.read(filename)
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform.T)
    if orig_sr != target_sr:
        waveform = resampy.resample(waveform, orig_sr, target_sr, filter=
            'kaiser_best')
    waveform = normalize_wav(waveform)[None, ...]
    waveform = pad_wav(waveform, segment_length)
    waveform = waveform / np.abs(waveform).max()
    return waveform * 0.5
