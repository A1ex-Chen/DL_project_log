def get_mid_block_adapter(base_channels: int, ctrl_channels: int,
    temb_channels: Optional[int]=None, max_norm_num_groups: Optional[int]=
    32, transformer_layers_per_block: int=1, num_attention_heads: Optional[
    int]=1, cross_attention_dim: Optional[int]=1024, upcast_attention: bool
    =False):
    base_to_ctrl = make_zero_conv(base_channels, base_channels)
    midblock = UNetMidBlock2DCrossAttn(transformer_layers_per_block=
        transformer_layers_per_block, in_channels=ctrl_channels +
        base_channels, out_channels=ctrl_channels, temb_channels=
        temb_channels, resnet_groups=find_largest_factor(gcd(ctrl_channels,
        ctrl_channels + base_channels), max_norm_num_groups),
        cross_attention_dim=cross_attention_dim, num_attention_heads=
        num_attention_heads, use_linear_projection=True, upcast_attention=
        upcast_attention)
    ctrl_to_base = make_zero_conv(ctrl_channels, base_channels)
    return MidBlockControlNetXSAdapter(base_to_ctrl=base_to_ctrl, midblock=
        midblock, ctrl_to_base=ctrl_to_base)
