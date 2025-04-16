def get_dummy_inputs(self, device, seed=0):
    np.random.seed(seed)
    reference_image = np.random.rand(32, 32, 3) * 255
    reference_image = Image.fromarray(reference_image.astype('uint8')).convert(
        'RGBA')
    cond_image = np.random.rand(32, 32, 3) * 255
    cond_image = Image.fromarray(cond_image.astype('uint8')).convert('RGBA')
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'swimming underwater', 'generator': generator,
        'reference_image': reference_image, 'condtioning_image': cond_image,
        'source_subject_category': 'dog', 'target_subject_category': 'dog',
        'height': 32, 'width': 32, 'guidance_scale': 7.5,
        'num_inference_steps': 2, 'output_type': 'np'}
    return inputs
