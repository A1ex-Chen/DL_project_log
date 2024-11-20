def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'horse', 'generator': generator,
        'prior_guidance_scale': 4.0, 'decoder_guidance_scale': 4.0,
        'num_inference_steps': 2, 'prior_num_inference_steps': 2,
        'output_type': 'np', 'height': 128, 'width': 128}
    return inputs
