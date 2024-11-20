def get_dummy_inputs_2images(self, device, seed=0, img_res=64):
    image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)
        ).to(device)
    image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed +
        22)).to(device)
    init_image1 = 2.0 * image1 - 1.0
    init_image2 = 2.0 * image2 - 1.0
    mask_image = torch.zeros((1, 1, img_res, img_res), device=device)
    if str(device).startswith('mps'):
        generator1 = torch.manual_seed(seed)
        generator2 = torch.manual_seed(seed)
    else:
        generator1 = torch.Generator(device=device).manual_seed(seed)
        generator2 = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': ['A painting of a squirrel eating a burger'] * 2,
        'image': [init_image1, init_image2], 'mask_image': [mask_image] * 2,
        'generator': [generator1, generator2], 'num_inference_steps': 2,
        'guidance_scale': 6.0, 'output_type': 'np'}
    return inputs
