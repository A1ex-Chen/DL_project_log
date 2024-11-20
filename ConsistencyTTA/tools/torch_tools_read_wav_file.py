def read_wav_file(filename, segment_length, target_sr=16000):
    wav, orig_sr = sf.read(filename)
    if len(wav.shape) > 1:
        wav = librosa.to_mono(wav.T)
    if not isinstance(target_sr, list):
        target_sr = [target_sr]
    cur_sr = orig_sr
    for tar_sr in target_sr:
        if cur_sr != tar_sr:
            wav = resampy.resample(wav, cur_sr, tar_sr, filter='kaiser_best')
            cur_sr = tar_sr
    wav = torch.from_numpy(wav)
    wav = wav - wav.mean()
    wav = wav / (wav.abs().max() + 1e-08) / 2
    wav = pad_wav(wav, segment_length).unsqueeze(0)
    wav = wav / (wav.abs().max() + 1e-08) / 2
    return wav.float()
