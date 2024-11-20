def generate(prompt: str, seed: int=None, cfg_weight: float=4.0):
    """ Generate audio from a given prompt.
    Args:
        prompt (str): Text prompt to generate audio from.
        seed (int, optional): Random seed. Defaults to None, which means no seed.
    """
    if seed is not None:
        seed_all(seed)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
            enabled=torch.cuda.is_available()):
            wav = consistencytta([prompt], num_steps=1, cfg_scale_input=
                cfg_weight, cfg_scale_post=1.0, sr=sr)
        sf.write('output.wav', wav.T, samplerate=sr, subtype='PCM_16')
    return 'output.wav'
