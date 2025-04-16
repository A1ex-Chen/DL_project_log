def get_inputs(self, device, generator_device='cpu', dtype=torch.float32,
    seed=0):
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    inputs = {'prompt': 'a fantasy landscape, concept art, high resolution',
        'generator': generator, 'num_inference_steps': 2, 'strength': 0.75,
        'guidance_scale': 7.5, 'output_type': 'np'}
    return inputs
