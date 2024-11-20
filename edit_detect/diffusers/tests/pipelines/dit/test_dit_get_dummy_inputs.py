def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'class_labels': [1], 'generator': generator,
        'num_inference_steps': 2, 'output_type': 'np'}
    return inputs
