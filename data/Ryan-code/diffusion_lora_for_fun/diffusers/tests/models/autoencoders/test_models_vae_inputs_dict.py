def inputs_dict(self, seed=None):
    generator = torch.Generator('cpu')
    if seed is not None:
        generator.manual_seed(0)
    image = randn_tensor((4, 3, 32, 32), generator=generator, device=torch.
        device(torch_device))
    return {'sample': image, 'generator': generator}
