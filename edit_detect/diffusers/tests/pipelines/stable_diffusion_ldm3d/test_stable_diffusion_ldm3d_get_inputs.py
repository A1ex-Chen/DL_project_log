def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    inputs = {'prompt': 'a photograph of an astronaut riding a horse',
        'latents': latents, 'generator': generator, 'num_inference_steps': 
        50, 'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs
