def get_dummy_inputs(self, device, seed=0):
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'batch_size': 1, 'num_inference_steps': None, 'timesteps': [
        22, 0], 'generator': generator, 'output_type': 'np'}
    return inputs
