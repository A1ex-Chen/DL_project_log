@property
def dummy_input(self, sizes=(32, 32)):
    batch_size = 4
    num_channels = 3
    image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    return {'sample': image}
