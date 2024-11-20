def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': '.', 'generator': generator, 'num_inference_steps':
        2, 'guidance_scale': 1.0, 'sag_scale': 1.0, 'output_type': 'np'}
    return inputs
