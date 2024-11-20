def get_inputs(self, generator_device='cpu', seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    inputs = {'prompt': 'a photograph of an astronaut riding a horse',
        'generator': generator, 'num_inference_steps': 50, 'guidance_scale':
        7.5, 'output_type': 'np'}
    return inputs
