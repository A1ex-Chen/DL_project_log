def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
    inputs = {'image': image, 'prompt':
        'A painting of a squirrel eating a burger', 'generator': generator,
        'num_inference_steps': 2, 'guidance_scale': 7.5, 'output_type': 'pt'}
    return inputs