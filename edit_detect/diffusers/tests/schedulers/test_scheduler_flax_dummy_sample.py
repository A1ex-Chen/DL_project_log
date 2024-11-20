@property
def dummy_sample(self):
    batch_size = 4
    num_channels = 3
    height = 8
    width = 8
    key1, key2 = random.split(random.PRNGKey(0))
    sample = random.uniform(key1, (batch_size, num_channels, height, width))
    return sample, key2
