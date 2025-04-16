def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'horse', 'generator': generator,
        'num_inference_steps': 1, 'frame_size': 32, 'output_type': 'latent'}
    return inputs
