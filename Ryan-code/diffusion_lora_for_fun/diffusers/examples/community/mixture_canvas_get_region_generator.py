def get_region_generator(self, device='cpu'):
    """Creates a torch.Generator based on the random seed of this region"""
    return torch.Generator(device).manual_seed(self.region_seed)
