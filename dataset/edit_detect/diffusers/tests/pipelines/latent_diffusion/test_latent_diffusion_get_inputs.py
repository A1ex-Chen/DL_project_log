def get_inputs(self, device, dtype=torch.float32, seed=0):
    generator = torch.manual_seed(seed)
    latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
    latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'latents': latents, 'generator': generator, 'num_inference_steps': 
        50, 'guidance_scale': 6.0, 'output_type': 'np'}
    return inputs
