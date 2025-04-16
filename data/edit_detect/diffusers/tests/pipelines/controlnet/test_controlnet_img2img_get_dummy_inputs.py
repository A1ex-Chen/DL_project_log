def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    controlnet_embedder_scale_factor = 2
    control_image = [randn_tensor((1, 3, 32 *
        controlnet_embedder_scale_factor, 32 *
        controlnet_embedder_scale_factor), generator=generator, device=
        torch.device(device)), randn_tensor((1, 3, 32 *
        controlnet_embedder_scale_factor, 32 *
        controlnet_embedder_scale_factor), generator=generator, device=
        torch.device(device))]
    image = floats_tensor(control_image[0].shape, rng=random.Random(seed)).to(
        device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    image = Image.fromarray(np.uint8(image)).convert('RGB').resize((64, 64))
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np', 'image': image, 'control_image':
        control_image}
    return inputs
