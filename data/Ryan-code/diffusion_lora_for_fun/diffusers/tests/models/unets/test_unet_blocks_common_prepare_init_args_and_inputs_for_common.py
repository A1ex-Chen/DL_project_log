def prepare_init_args_and_inputs_for_common(self):
    init_dict = {'in_channels': 32, 'out_channels': 32, 'temb_channels': 128}
    if self.block_type == 'up':
        init_dict['prev_output_channel'] = 32
    if self.block_type == 'mid':
        init_dict.pop('out_channels')
    inputs_dict = self.dummy_input
    return init_dict, inputs_dict
