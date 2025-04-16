@property
def dummy_input(self):
    batch_size = 4
    num_channels = 4
    sizes = 16, 16
    conditioning_image_size = 3, 32, 32
    noise = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor([10]).to(torch_device)
    encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(torch_device)
    controlnet_cond = floats_tensor((batch_size, *conditioning_image_size)).to(
        torch_device)
    conditioning_scale = 1
    return {'sample': noise, 'timestep': time_step, 'encoder_hidden_states':
        encoder_hidden_states, 'controlnet_cond': controlnet_cond,
        'conditioning_scale': conditioning_scale}
