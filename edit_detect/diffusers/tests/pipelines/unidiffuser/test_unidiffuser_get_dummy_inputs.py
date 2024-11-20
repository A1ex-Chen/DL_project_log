def get_dummy_inputs(self, device, seed=0):
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'an elephant under the sea', 'image': image,
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np'}
    return inputs
