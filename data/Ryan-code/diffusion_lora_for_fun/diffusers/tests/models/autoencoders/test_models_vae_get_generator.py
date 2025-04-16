def get_generator(self, seed=0):
    generator_device = 'cpu' if not torch_device.startswith('cuda') else 'cuda'
    if torch_device != 'mps':
        return torch.Generator(device=generator_device).manual_seed(seed)
    return torch.manual_seed(seed)
