@property
def dummy_sample_deter(self):
    batch_size = 4
    num_channels = 3
    height = 8
    width = 8
    num_elems = batch_size * num_channels * height * width
    sample = torch.arange(num_elems)
    sample = sample.reshape(num_channels, height, width, batch_size)
    sample = sample / num_elems
    sample = sample.permute(3, 0, 1, 2)
    return sample
