def get_down_block_adapter(base_in_channels: int, base_out_channels: int,
    ctrl_in_channels: int, ctrl_out_channels: int, temb_channels: int,
    max_norm_num_groups: Optional[int]=32, has_crossattn=True,
    transformer_layers_per_block: Optional[Union[int, Tuple[int]]]=1,
    num_attention_heads: Optional[int]=1, cross_attention_dim: Optional[int
    ]=1024, add_downsample: bool=True, upcast_attention: Optional[bool]=False):
    num_layers = 2
    resnets = []
    attentions = []
    ctrl_to_base = []
    base_to_ctrl = []
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block
            ] * num_layers
    for i in range(num_layers):
        base_in_channels = base_in_channels if i == 0 else base_out_channels
        ctrl_in_channels = ctrl_in_channels if i == 0 else ctrl_out_channels
        base_to_ctrl.append(make_zero_conv(base_in_channels, base_in_channels))
        resnets.append(ResnetBlock2D(in_channels=ctrl_in_channels +
            base_in_channels, out_channels=ctrl_out_channels, temb_channels
            =temb_channels, groups=find_largest_factor(ctrl_in_channels +
            base_in_channels, max_factor=max_norm_num_groups), groups_out=
            find_largest_factor(ctrl_out_channels, max_factor=
            max_norm_num_groups), eps=1e-05))
        if has_crossattn:
            attentions.append(Transformer2DModel(num_attention_heads, 
                ctrl_out_channels // num_attention_heads, in_channels=
                ctrl_out_channels, num_layers=transformer_layers_per_block[
                i], cross_attention_dim=cross_attention_dim,
                use_linear_projection=True, upcast_attention=
                upcast_attention, norm_num_groups=find_largest_factor(
                ctrl_out_channels, max_factor=max_norm_num_groups)))
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels,
            base_out_channels))
    if add_downsample:
        base_to_ctrl.append(make_zero_conv(base_out_channels,
            base_out_channels))
        downsamplers = Downsample2D(ctrl_out_channels + base_out_channels,
            use_conv=True, out_channels=ctrl_out_channels, name='op')
        ctrl_to_base.append(make_zero_conv(ctrl_out_channels,
            base_out_channels))
    else:
        downsamplers = None
    down_block_components = DownBlockControlNetXSAdapter(resnets=nn.
        ModuleList(resnets), base_to_ctrl=nn.ModuleList(base_to_ctrl),
        ctrl_to_base=nn.ModuleList(ctrl_to_base))
    if has_crossattn:
        down_block_components.attentions = nn.ModuleList(attentions)
    if downsamplers is not None:
        down_block_components.downsamplers = downsamplers
    return down_block_components
