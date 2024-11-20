@classmethod
def from_modules(cls, base_upblock: CrossAttnUpBlock2D, ctrl_upblock:
    UpBlockControlNetXSAdapter):
    ctrl_to_base_skip_connections = ctrl_upblock.ctrl_to_base

    def get_first_cross_attention(block):
        return block.attentions[0].transformer_blocks[0].attn2
    out_channels = base_upblock.resnets[0].out_channels
    in_channels = base_upblock.resnets[-1].in_channels - out_channels
    prev_output_channels = base_upblock.resnets[0].in_channels - out_channels
    ctrl_skip_channelss = [c.in_channels for c in ctrl_to_base_skip_connections
        ]
    temb_channels = base_upblock.resnets[0].time_emb_proj.in_features
    num_groups = base_upblock.resnets[0].norm1.num_groups
    resolution_idx = base_upblock.resolution_idx
    if hasattr(base_upblock, 'attentions'):
        has_crossattn = True
        transformer_layers_per_block = len(base_upblock.attentions[0].
            transformer_blocks)
        num_attention_heads = get_first_cross_attention(base_upblock).heads
        cross_attention_dim = get_first_cross_attention(base_upblock
            ).cross_attention_dim
        upcast_attention = get_first_cross_attention(base_upblock
            ).upcast_attention
    else:
        has_crossattn = False
        transformer_layers_per_block = None
        num_attention_heads = None
        cross_attention_dim = None
        upcast_attention = None
    add_upsample = base_upblock.upsamplers is not None
    model = cls(in_channels=in_channels, out_channels=out_channels,
        prev_output_channel=prev_output_channels, ctrl_skip_channels=
        ctrl_skip_channelss, temb_channels=temb_channels, norm_num_groups=
        num_groups, resolution_idx=resolution_idx, has_crossattn=
        has_crossattn, transformer_layers_per_block=
        transformer_layers_per_block, num_attention_heads=
        num_attention_heads, cross_attention_dim=cross_attention_dim,
        add_upsample=add_upsample, upcast_attention=upcast_attention)
    model.resnets.load_state_dict(base_upblock.resnets.state_dict())
    if has_crossattn:
        model.attentions.load_state_dict(base_upblock.attentions.state_dict())
    if add_upsample:
        model.upsamplers.load_state_dict(base_upblock.upsamplers[0].
            state_dict())
    model.ctrl_to_base.load_state_dict(ctrl_to_base_skip_connections.
        state_dict())
    return model
