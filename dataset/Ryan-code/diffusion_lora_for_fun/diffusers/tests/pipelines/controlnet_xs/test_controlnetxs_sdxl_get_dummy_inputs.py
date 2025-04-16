def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    controlnet_embedder_scale_factor = 2
    image = randn_tensor((1, 3, 8 * controlnet_embedder_scale_factor, 8 *
        controlnet_embedder_scale_factor), generator=generator, device=
        torch.device(device))
    inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'generator': generator, 'num_inference_steps': 2, 'guidance_scale':
        6.0, 'output_type': 'np', 'image': image}
    return inputs
