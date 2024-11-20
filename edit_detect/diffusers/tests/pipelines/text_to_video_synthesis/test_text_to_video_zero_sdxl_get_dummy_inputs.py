def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'prompt': 'A panda dancing in Antarctica', 'generator':
        generator, 'num_inference_steps': 5, 't0': 1, 't1': 3, 'height': 64,
        'width': 64, 'video_length': 3, 'output_type': 'np'}
    return inputs
