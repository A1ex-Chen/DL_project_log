def pad_short_audio(audio, min_samples=32000):
    if audio.shape[-1] < min_samples:
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.
            shape[-1]), mode='constant', value=0.0)
    return audio
