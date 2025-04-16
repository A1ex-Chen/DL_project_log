def get_dummy_inputs(self, device, seed=0):
    input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(
        device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'image': input_image, 'generator': generator,
        'num_inference_steps': 1, 'frame_size': 32, 'output_type': 'latent'}
    return inputs
