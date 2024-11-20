def read_centered_wav(audio_file, target_sr):
    audio, orig_sr = sf.read(audio_file)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.T)
    if orig_sr != target_sr and orig_sr % target_sr == 0:
        audio = audio[..., ::orig_sr // target_sr]
    elif orig_sr != target_sr:
        audio = resampy.resample(audio, orig_sr, target_sr, filter=
            'kaiser_best')
    return audio - audio.mean()
