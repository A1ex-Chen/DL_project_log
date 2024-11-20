def get_dummy_inputs(self, device, seed=0):
    mask = floats_tensor((1, 16, 16), rng=random.Random(seed)).to(device)
    latents = floats_tensor((1, 2, 4, 16, 16), rng=random.Random(seed)).to(
        device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'a dog and a newt', 'mask_image': mask,
        'image_latents': latents, 'generator': generator,
        'num_inference_steps': 2, 'inpaint_strength': 1.0, 'guidance_scale':
        6.0, 'output_type': 'np'}
    return inputs
