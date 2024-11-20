def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    image = torch.full((1, 3, 4, 4), 1.0, dtype=torch.float32, device=device)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'output_type':
        'np', 'image': image}
    return inputs
