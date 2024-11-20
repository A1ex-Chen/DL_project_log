@property
def dummy_input(self):
    batch_size = 4
    num_channels = 4
    sizes = 16, 16
    noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)
    return {'sample': noise, 'timestep': time_step, 'encoder_hidden_states':
        encoder_hidden_states}
