def __init__(self, in_channels: int, out_channels: int, num_blocks: Tuple[
    int, ...], block_out_channels: Tuple[int, ...],
    upsampling_scaling_factor: int, act_fn: str, upsample_fn: str):
    super().__init__()
    layers = [nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3,
        padding=1), get_activation(act_fn)]
    for i, num_block in enumerate(num_blocks):
        is_final_block = i == len(num_blocks) - 1
        num_channels = block_out_channels[i]
        for _ in range(num_block):
            layers.append(AutoencoderTinyBlock(num_channels, num_channels,
                act_fn))
        if not is_final_block:
            layers.append(nn.Upsample(scale_factor=
                upsampling_scaling_factor, mode=upsample_fn))
        conv_out_channel = num_channels if not is_final_block else out_channels
        layers.append(nn.Conv2d(num_channels, conv_out_channel, kernel_size
            =3, padding=1, bias=is_final_block))
    self.layers = nn.Sequential(*layers)
    self.gradient_checkpointing = False
