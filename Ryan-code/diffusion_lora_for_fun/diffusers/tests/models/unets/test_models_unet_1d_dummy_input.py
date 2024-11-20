@property
def dummy_input(self):
    batch_size = 4
    num_features = 14
    seq_len = 16
    noise = floats_tensor((batch_size, num_features, seq_len)).to(torch_device)
    time_step = torch.tensor([10] * batch_size).to(torch_device)
    return {'sample': noise, 'timestep': time_step}
