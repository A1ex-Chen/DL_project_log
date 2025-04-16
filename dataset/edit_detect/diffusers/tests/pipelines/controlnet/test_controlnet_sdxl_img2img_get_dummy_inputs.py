def get_dummy_inputs(self, device, seed=0):
    controlnet_embedder_scale_factor = 2
    image = floats_tensor((1, 3, 32 * controlnet_embedder_scale_factor, 32 *
        controlnet_embedder_scale_factor), rng=random.Random(seed)).to(device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np', 'image': image, 'control_image': image}
    return inputs
