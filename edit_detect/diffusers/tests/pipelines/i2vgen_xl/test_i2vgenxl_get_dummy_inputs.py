def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(
        device)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'image':
        input_image, 'generator': generator, 'num_inference_steps': 2,
        'guidance_scale': 6.0, 'output_type': 'pt', 'num_frames': 4,
        'width': 32, 'height': 32}
    return inputs
