def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A modern livingroom', 'generator': generator,
        'num_inference_steps': 2, 'guidance_scale': 6.0, 'gligen_phrases':
        ['a birthday cake'], 'gligen_boxes': [[0.2676, 0.6088, 0.4773, 
        0.7183]], 'output_type': 'np'}
    return inputs
