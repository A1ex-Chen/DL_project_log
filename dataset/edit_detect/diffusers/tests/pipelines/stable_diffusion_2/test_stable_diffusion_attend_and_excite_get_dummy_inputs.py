def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = inputs = {'prompt': 'a cat and a frog', 'token_indices': [2, 5
        ], 'generator': generator, 'num_inference_steps': 1,
        'guidance_scale': 6.0, 'output_type': 'np', 'max_iter_to_alter': 2,
        'thresholds': {(0): 0.7}}
    return inputs
