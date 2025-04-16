@property
def dummy_sample(self):
    batch_size = 4
    num_channels = 3
    height = 8
    width = 8
    sample = torch.rand((batch_size, num_channels, height, width))
    return sample
