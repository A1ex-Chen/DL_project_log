def get_inputs_tts(self, device, generator_device='cpu', dtype=torch.
    float32, seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    latents = np.random.RandomState(seed).standard_normal((1, 8, 128, 16))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    inputs = {'prompt': 'A men saying', 'transcription':
        'hello my name is John', 'latents': latents, 'generator': generator,
        'num_inference_steps': 3, 'guidance_scale': 2.5}
    return inputs
