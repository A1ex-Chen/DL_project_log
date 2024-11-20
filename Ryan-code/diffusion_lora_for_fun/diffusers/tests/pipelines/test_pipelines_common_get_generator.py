def get_generator(self, seed):
    device = torch_device if torch_device != 'mps' else 'cpu'
    generator = torch.Generator(device).manual_seed(seed)
    return generator
