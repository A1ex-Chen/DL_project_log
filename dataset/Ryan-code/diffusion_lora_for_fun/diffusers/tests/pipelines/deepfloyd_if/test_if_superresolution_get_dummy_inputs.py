def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'image':
        image, 'generator': generator, 'num_inference_steps': 2,
        'output_type': 'np'}
    return inputs
