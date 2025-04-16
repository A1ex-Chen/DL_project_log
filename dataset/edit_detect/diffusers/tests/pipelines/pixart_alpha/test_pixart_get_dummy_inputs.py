def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        5.0, 'use_resolution_binning': False, 'output_type': 'np'}
    return inputs
