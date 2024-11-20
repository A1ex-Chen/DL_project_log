def get_dummy_inputs(self, device, seed=0):
    image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    init_image = Image.fromarray(np.uint8(image)).convert('RGB')
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'image':
        init_image, 'generator': generator, 'strength': 0.75,
        'num_inference_steps': 10, 'guidance_scale': 6.0, 'output_type': 'np'}
    return inputs
