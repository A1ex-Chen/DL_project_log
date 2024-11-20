def get_dummy_input(self, include_temb=True,
    include_res_hidden_states_tuple=False, include_encoder_hidden_states=
    False, include_skip_sample=False):
    batch_size = 4
    num_channels = 32
    sizes = 32, 32
    generator = torch.manual_seed(0)
    device = torch.device(torch_device)
    shape = (batch_size, num_channels) + sizes
    hidden_states = randn_tensor(shape, generator=generator, device=device)
    dummy_input = {'hidden_states': hidden_states}
    if include_temb:
        temb_channels = 128
        dummy_input['temb'] = randn_tensor((batch_size, temb_channels),
            generator=generator, device=device)
    if include_res_hidden_states_tuple:
        generator_1 = torch.manual_seed(1)
        dummy_input['res_hidden_states_tuple'] = randn_tensor(shape,
            generator=generator_1, device=device),
    if include_encoder_hidden_states:
        dummy_input['encoder_hidden_states'] = floats_tensor((batch_size, 
            32, 32)).to(torch_device)
    if include_skip_sample:
        dummy_input['skip_sample'] = randn_tensor((batch_size, 3) + sizes,
            generator=generator, device=device)
    return dummy_input
