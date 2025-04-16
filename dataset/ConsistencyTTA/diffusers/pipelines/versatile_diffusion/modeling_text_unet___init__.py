def __init__(self, in_channels: int, temb_channels: int, dropout: float=0.0,
    num_layers: int=1, resnet_eps: float=1e-06, resnet_time_scale_shift:
    str='default', resnet_act_fn: str='swish', resnet_groups: int=32,
    resnet_pre_norm: bool=True, attn_num_head_channels=1,
    output_scale_factor=1.0, cross_attention_dim=1280):
    super().__init__()
    self.has_cross_attention = True
    self.attn_num_head_channels = attn_num_head_channels
    resnet_groups = resnet_groups if resnet_groups is not None else min(
        in_channels // 4, 32)
    self.num_heads = in_channels // self.attn_num_head_channels
    resnets = [ResnetBlockFlat(in_channels=in_channels, out_channels=
        in_channels, temb_channels=temb_channels, eps=resnet_eps, groups=
        resnet_groups, dropout=dropout, time_embedding_norm=
        resnet_time_scale_shift, non_linearity=resnet_act_fn,
        output_scale_factor=output_scale_factor, pre_norm=resnet_pre_norm)]
    attentions = []
    for _ in range(num_layers):
        attentions.append(Attention(query_dim=in_channels,
            cross_attention_dim=in_channels, heads=self.num_heads, dim_head
            =attn_num_head_channels, added_kv_proj_dim=cross_attention_dim,
            norm_num_groups=resnet_groups, bias=True, upcast_softmax=True,
            processor=AttnAddedKVProcessor()))
        resnets.append(ResnetBlockFlat(in_channels=in_channels,
            out_channels=in_channels, temb_channels=temb_channels, eps=
            resnet_eps, groups=resnet_groups, dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift, non_linearity=
            resnet_act_fn, output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm))
    self.attentions = nn.ModuleList(attentions)
    self.resnets = nn.ModuleList(resnets)
