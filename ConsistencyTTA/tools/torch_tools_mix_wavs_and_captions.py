def mix_wavs_and_captions(wave1, wave2, caption1, caption2):
    mixed_sound = mix(wave1, wave2, 0.5, 16000).reshape(1, -1)
    mixed_caption = f'{caption1} and {uncapitalize(caption2)}'
    return mixed_sound, mixed_caption
