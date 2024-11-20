def load_audio_task(fname, target_sr=16000, target_length=1000):
    wav_data, orig_sr = sf.read(fname, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    wav_data = wav_data / 32768.0
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    if orig_sr % target_sr == 0:
        wav_data = wav_data[::orig_sr // target_sr]
    elif orig_sr != target_sr:
        wav_data = resampy.resample(wav_data, orig_sr, target_sr, filter=
            'kaiser_best')
    return wav_data[:int(target_length * target_sr / 100)]
