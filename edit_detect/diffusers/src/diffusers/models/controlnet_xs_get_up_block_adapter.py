def get_up_block_adapter(out_channels: int, prev_output_channel: int,
    ctrl_skip_channels: List[int]):
    ctrl_to_base = []
    num_layers = 3
    for i in range(num_layers):
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i],
            resnet_in_channels))
    return UpBlockControlNetXSAdapter(ctrl_to_base=nn.ModuleList(ctrl_to_base))
