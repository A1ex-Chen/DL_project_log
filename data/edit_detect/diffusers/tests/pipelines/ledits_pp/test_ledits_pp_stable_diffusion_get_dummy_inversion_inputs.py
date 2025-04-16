def get_dummy_inversion_inputs(self, device, seed=0):
    images = floats_tensor((2, 3, 32, 32), rng=random.Random(0)).cpu().permute(
        0, 2, 3, 1)
    images = 255 * images
    image_1 = Image.fromarray(np.uint8(images[0])).convert('RGB')
    image_2 = Image.fromarray(np.uint8(images[1])).convert('RGB')
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'image': [image_1, image_2], 'source_prompt': '',
        'source_guidance_scale': 3.5, 'num_inversion_steps': 20, 'skip': 
        0.15, 'generator': generator}
    return inputs
