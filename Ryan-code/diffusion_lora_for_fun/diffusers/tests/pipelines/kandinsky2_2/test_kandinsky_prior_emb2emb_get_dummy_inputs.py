def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    init_image = Image.fromarray(np.uint8(image)).convert('RGB').resize((
        256, 256))
    inputs = {'prompt': 'horse', 'image': init_image, 'strength': 0.5,
        'generator': generator, 'guidance_scale': 4.0,
        'num_inference_steps': 2, 'output_type': 'np'}
    return inputs
