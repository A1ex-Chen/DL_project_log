def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    latents = np.random.RandomState(seed).standard_normal((1, 8, 128, 16))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    inputs = {'prompt': 'A hammer hitting a wooden surface', 'latents':
        latents, 'generator': generator, 'num_inference_steps': 3,
        'guidance_scale': 2.5}
    return inputs
