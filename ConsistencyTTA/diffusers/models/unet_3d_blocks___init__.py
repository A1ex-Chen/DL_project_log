def __init__(self, in_channels: int, prev_output_channel: int, out_channels:
    int, temb_channels: int, dropout: float=0.0, num_layers: int=1,
    resnet_eps: float=1e-06, resnet_time_scale_shift: str='default',
    resnet_act_fn: str='swish', resnet_groups: int=32, resnet_pre_norm:
    bool=True, output_scale_factor=1.0, add_upsample=True):
    super().__init__()
    resnets = []
    temp_convs = []
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
        temp_convs.append(TemporalConvLayer(out_channels, out_channels,
            dropout=0.1))
    self.resnets = nn.ModuleList(resnets)
    self.temp_convs = nn.ModuleList(temp_convs)
    if add_upsample:
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=
            True, out_channels=out_channels)])
    else:
        self.upsamplers = None
    self.gradient_checkpointing = False
