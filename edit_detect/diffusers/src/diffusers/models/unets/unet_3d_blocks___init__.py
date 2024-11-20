def __init__(self, in_channels: int, out_channels: int, prev_output_channel:
    int, temb_channels: int, resolution_idx: Optional[int]=None, num_layers:
    int=1, transformer_layers_per_block: Union[int, Tuple[int]]=1,
    resnet_eps: float=1e-06, num_attention_heads: int=1,
    cross_attention_dim: int=1280, add_upsample: bool=True):
    super().__init__()
    resnets = []
    attentions = []
    self.has_cross_attention = True
    self.num_attention_heads = num_attention_heads
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block
            ] * num_layers
    for i in range(num_layers):
        res_skip_channels = (in_channels if i == num_layers - 1 else
            out_channels)
        resnet_in_channels = prev_output_channel if i == 0 else out_channels
        resnets.append(SpatioTemporalResBlock(in_channels=
            resnet_in_channels + res_skip_channels, out_channels=
            out_channels, temb_channels=temb_channels, eps=resnet_eps))
        attentions.append(TransformerSpatioTemporalModel(
            num_attention_heads, out_channels // num_attention_heads,
            in_channels=out_channels, num_layers=
            transformer_layers_per_block[i], cross_attention_dim=
            cross_attention_dim))
    self.attentions = nn.ModuleList(attentions)
    self.resnets = nn.ModuleList(resnets)
    if add_upsample:
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=
            True, out_channels=out_channels)])
    else:
        self.upsamplers = None
    self.gradient_checkpointing = False
    self.resolution_idx = resolution_idx
