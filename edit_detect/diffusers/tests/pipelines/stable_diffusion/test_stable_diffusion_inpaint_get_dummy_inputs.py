def get_dummy_inputs(self, device, seed=0, img_res=64, output_pil=True):
    if output_pil:
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(
            device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        mask_image = torch.ones_like(image)
        image = 255 * image
        mask_image = 255 * mask_image
        init_image = Image.fromarray(np.uint8(image)).convert('RGB').resize((
            img_res, img_res))
        mask_image = Image.fromarray(np.uint8(mask_image)).convert('RGB'
            ).resize((img_res, img_res))
    else:
        image = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)
            ).to(device)
        init_image = 2.0 * image - 1.0
        mask_image = torch.ones((1, 1, img_res, img_res), device=device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A painting of a squirrel eating a burger', 'image':
        init_image, 'mask_image': mask_image, 'generator': generator,
        'num_inference_steps': 2, 'guidance_scale': 6.0, 'output_type': 'np'}
    return inputs
