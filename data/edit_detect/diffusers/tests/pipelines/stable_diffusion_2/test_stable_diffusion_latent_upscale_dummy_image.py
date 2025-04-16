@property
def dummy_image(self):
    batch_size = 1
    num_channels = 4
    sizes = 16, 16
    image = floats_tensor((batch_size, num_channels) + sizes, rng=random.
        Random(0)).to(torch_device)
    return image
