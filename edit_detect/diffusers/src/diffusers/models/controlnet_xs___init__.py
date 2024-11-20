def __init__(self, in_channels: int, out_channels: int, prev_output_channel:
    int, ctrl_skip_channels: List[int], temb_channels: int, norm_num_groups:
    int=32, resolution_idx: Optional[int]=None, has_crossattn=True,
    transformer_layers_per_block: int=1, num_attention_heads: int=1,
    cross_attention_dim: int=1024, add_upsample: bool=True,
    upcast_attention: bool=False):
    super().__init__()
    resnets = []
    attentions = []
    ctrl_to_base = []
    num_layers = 3
    self.has_cross_attention = has_crossattn
    self.num_attention_heads = num_attention_heads
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block
            ] * num_layers
    for i in range(num_layers):
        res_skip_channels = (in_channels if i == num_layers - 1 else
            out_channels)
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        ctrl_to_base.append(make_zero_conv(ctrl_skip_channels[i],
            resnet_in_channels))
        resnets.append(ResnetBlock2D(in_channels=resnet_in_channels +
            res_skip_channels, out_channels=out_channels, temb_channels=
            temb_channels, groups=norm_num_groups))
        if has_crossattn:
            attentions.append(Transformer2DModel(num_attention_heads, 
                out_channels // num_attention_heads, in_channels=
                out_channels, num_layers=transformer_layers_per_block[i],
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=True, upcast_attention=
                upcast_attention, norm_num_groups=norm_num_groups))
    self.resnets = nn.ModuleList(resnets)
    self.attentions = nn.ModuleList(attentions) if has_crossattn else [None
        ] * num_layers
    self.ctrl_to_base = nn.ModuleList(ctrl_to_base)
    if add_upsample:
        self.upsamplers = Upsample2D(out_channels, use_conv=True,
            out_channels=out_channels)
    else:
        self.upsamplers = None
    self.gradient_checkpointing = False
    self.resolution_idx = resolution_idx
