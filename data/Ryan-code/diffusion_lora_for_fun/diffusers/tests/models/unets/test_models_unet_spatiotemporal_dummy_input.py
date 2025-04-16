@property
def dummy_input(self):
    batch_size = 2
    num_frames = 2
    num_channels = 4
    sizes = 32, 32
    noise = floats_tensor((batch_size, num_frames, num_channels) + sizes).to(
        torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = floats_tensor((batch_size, 1, 32)).to(torch_device)
    return {'sample': noise, 'timestep': time_step, 'encoder_hidden_states':
        encoder_hidden_states, 'added_time_ids': self._get_add_time_ids()}
