def __init__(self, in_channels: int, out_channels: int, prev_output_channel:
    int, temb_channels: int, dropout: float=0.0, num_layers: int=1,
    transformer_layers_per_block: int=1, resnet_eps: float=1e-06,
    resnet_time_scale_shift: str='default', resnet_act_fn: str='swish',
    resnet_groups: int=32, resnet_pre_norm: bool=True, num_attention_heads=
    1, cross_attention_dim=1280, output_scale_factor=1.0, add_upsample=True,
    use_linear_projection=False, only_cross_attention=False,
    upcast_attention=False):
    super().__init__()
    resnets = []
    attentions = []
    self.has_cross_attention = True
    self.num_attention_heads = num_attention_heads
    if isinstance(cross_attention_dim, int):
        cross_attention_dim = cross_attention_dim,
    if isinstance(cross_attention_dim, (list, tuple)) and len(
        cross_attention_dim) > 4:
        raise ValueError(
            f'Only up to 4 cross-attention layers are supported. Ensure that the length of cross-attention dims is less than or equal to 4. Got cross-attention dims {cross_attention_dim} of length {len(cross_attention_dim)}'
            )
    self.cross_attention_dim = cross_attention_dim
    for i in range(num_layers):
        res_skip_channels = (in_channels if i == num_layers - 1 else
            out_channels)
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        resnets.append(ResnetBlock2D(in_channels=resnet_in_channels +
            res_skip_channels, out_channels=out_channels, temb_channels=
            temb_channels, eps=resnet_eps, groups=resnet_groups, dropout=
            dropout, time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn, output_scale_factor=
            output_scale_factor, pre_norm=resnet_pre_norm))
        for j in range(len(cross_attention_dim)):
            attentions.append(Transformer2DModel(num_attention_heads, 
                out_channels // num_attention_heads, in_channels=
                out_channels, num_layers=transformer_layers_per_block,
                cross_attention_dim=cross_attention_dim[j], norm_num_groups
                =resnet_groups, use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention, upcast_attention
                =upcast_attention, double_self_attention=True if 
                cross_attention_dim[j] is None else False))
    self.attentions = nn.ModuleList(attentions)
    self.resnets = nn.ModuleList(resnets)
    if add_upsample:
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=
            True, out_channels=out_channels)])
    else:
        self.upsamplers = None
    self.gradient_checkpointing = False
