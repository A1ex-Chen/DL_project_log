def get_dummy_inputs(self, device, seed=0):
    video = floats_tensor((1, 3, 3, 32, 32), rng=random.Random(seed)).to(device
        )
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'video':
        video, 'generator': generator, 'num_inference_steps': 2,
        'guidance_scale': 6.0, 'output_type': 'pt'}
    return inputs
