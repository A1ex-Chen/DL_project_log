def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device='cpu').manual_seed(seed)
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(device)
    inputs = {'generator': generator, 'image': image, 'num_inference_steps':
        2, 'output_type': 'pt', 'min_guidance_scale': 1.0,
        'max_guidance_scale': 2.5, 'num_frames': 2, 'height': 32, 'width': 32}
    return inputs
