def get_dummy_inputs(self, device, seed=0, img_res=64):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    mask_image = torch.ones_like(image)
    controlnet_embedder_scale_factor = 2
    control_image = floats_tensor((1, 3, 32 *
        controlnet_embedder_scale_factor, 32 *
        controlnet_embedder_scale_factor), rng=random.Random(seed)).to(device
        ).cpu()
    control_image = control_image.cpu().permute(0, 2, 3, 1)[0]
    image = 255 * image
    mask_image = 255 * mask_image
    control_image = 255 * control_image
    init_image = Image.fromarray(np.uint8(image)).convert('RGB').resize((
        img_res, img_res))
    mask_image = Image.fromarray(np.uint8(mask_image)).convert('L').resize((
        img_res, img_res))
    control_image = Image.fromarray(np.uint8(control_image)).convert('RGB'
        ).resize((img_res, img_res))
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np', 'image': init_image, 'mask_image':
        mask_image, 'control_image': control_image}
    return inputs
