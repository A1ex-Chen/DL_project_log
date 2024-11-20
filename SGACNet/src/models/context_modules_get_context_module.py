def get_context_module(context_module_name, channels_in, channels_out,
    input_size, activation, upsampling_mode='bilinear'):
    if 'appm' in context_module_name:
        if context_module_name == 'appm-1-2-4-8':
            bins = 1, 2, 4, 8
        else:
            bins = 1, 5
        context_module = AdaptivePyramidPoolingModule(channels_in,
            channels_out, bins=bins, input_size=input_size, activation=
            activation, upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'ppm' in context_module_name:
        if context_module_name == 'ppm-1-2-4-8':
            bins = 1, 2, 4, 8
        else:
            bins = 1, 5
        context_module = PyramidPoolingModule(channels_in, channels_out,
            bins=bins, activation=activation, upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'sppm' in context_module_name:
        if context_module_name == 'sppm-1-3':
            bins = 1, 3
        else:
            bins = 1, 2, 4
        context_module = PPContextModule(channels_in, channels_out, bins=
            bins, upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'dappm' in context_module_name:
        context_module = DAPPM(channels_in, channels_out, activation=
            activation, upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    else:
        context_module = nn.Identity()
        channels_after_context_module = channels_in
    return context_module, channels_after_context_module
