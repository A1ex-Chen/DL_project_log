def get_dummy_inputs(self, device, seed=0, pil_image=True):
    input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(
        device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    if pil_image:
        input_image = input_image * 0.5 + 0.5
        input_image = input_image.clamp(0, 1)
        input_image = input_image.cpu().permute(0, 2, 3, 1).float().numpy()
        input_image = DiffusionPipeline.numpy_to_pil(input_image)[0]
    return {'image': input_image, 'generator': generator,
        'decoder_num_inference_steps': 2, 'super_res_num_inference_steps': 
        2, 'output_type': 'np'}
