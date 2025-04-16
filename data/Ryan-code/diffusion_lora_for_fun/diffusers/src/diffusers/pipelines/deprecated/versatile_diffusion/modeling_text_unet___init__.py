def __init__(self, in_channels: int, temb_channels: int, dropout: float=0.0,
    num_layers: int=1, resnet_eps: float=1e-06, resnet_time_scale_shift:
    str='default', resnet_act_fn: str='swish', resnet_groups: int=32,
    resnet_pre_norm: bool=True, attention_head_dim: int=1,
    output_scale_factor: float=1.0, cross_attention_dim: int=1280,
    skip_time_act: bool=False, only_cross_attention: bool=False,
    cross_attention_norm: Optional[str]=None):
    super().__init__()
    self.has_cross_attention = True
    self.attention_head_dim = attention_head_dim
    resnet_groups = resnet_groups if resnet_groups is not None else min(
        in_channels // 4, 32)
    self.num_heads = in_channels // self.attention_head_dim
    resnets = [ResnetBlockFlat(in_channels=in_channels, out_channels=
        in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=
        resnet_groups, dropout=dropout, time_embedding_norm=
        resnet_time_scale_shift, non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm,
        skip_time_act=skip_time_act)]
    attentions = []
    for _ in range(num_layers):
        processor = AttnAddedKVProcessor2_0() if hasattr(F,
            'scaled_dot_product_attention') else AttnAddedKVProcessor()
        attentions.append(Attention(query_dim=in_channels,
            cross_attention_dim=in_channels, heads=self.num_heads, dim_head
            =self.attention_head_dim, added_kv_proj_dim=cross_attention_dim,
            norm_num_groups=resnet_groups, bias=True, upcast_softmax=True,
            only_cross_attention=only_cross_attention, cross_attention_norm
            =cross_attention_norm, processor=processor))
        resnets.append(ResnetBlockFlat(in_channels=in_channels,
            out_channels=in_channels, temb_channels=temb_channels, eps=
            resnet_eps, groups=resnet_groups, dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift, non_linearity=
            resnet_act_fn, output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm, skip_time_act=skip_time_act))
    self.attentions = nn.ModuleList(attentions)
    self.resnets = nn.ModuleList(resnets)
