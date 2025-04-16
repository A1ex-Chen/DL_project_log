def get_dummy_inputs(self, device, seed=0):
    generator_device = 'cpu' if not device.startswith('cuda') else 'cuda'
    if not str(device).startswith('mps'):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
    else:
        generator = torch.manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np'}
    return inputs
