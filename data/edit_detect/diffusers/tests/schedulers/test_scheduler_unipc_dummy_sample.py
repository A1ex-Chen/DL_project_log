@property
def dummy_sample(self):
    batch_size = 4
    num_channels = 3
    width = 8
    sample = torch.rand((batch_size, num_channels, width))
    return sample
