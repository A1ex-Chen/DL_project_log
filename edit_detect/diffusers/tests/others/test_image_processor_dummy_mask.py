@property
def dummy_mask(self):
    batch_size = 1
    num_channels = 1
    height = 8
    width = 8
    sample = torch.rand((batch_size, num_channels, height, width))
    return sample
