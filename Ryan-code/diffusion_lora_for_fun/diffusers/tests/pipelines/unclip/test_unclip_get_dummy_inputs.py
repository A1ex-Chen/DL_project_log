def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'horse', 'generator': generator,
        'prior_num_inference_steps': 2, 'decoder_num_inference_steps': 2,
        'super_res_num_inference_steps': 2, 'output_type': 'np'}
    return inputs
