@property
def dummy_input(self, sizes=(32, 32)):
    batch_size = 4
    num_channels = 3
    noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor(batch_size * [10]).to(dtype=torch.int32,
        device=torch_device)
    return {'sample': noise, 'timestep': time_step}
