@property
def dummy_input(self):
    batch_size = 3
    num_channels = 3
    sizes = 32, 32
    image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    num_frames = 3
    return {'sample': image, 'num_frames': num_frames}
