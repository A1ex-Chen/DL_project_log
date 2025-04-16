def get_block(block_type, in_channels, nhead, c_skip=0, dropout=0,
    self_attn=True):
    if block_type == 'SDCascadeResBlock':
        return SDCascadeResBlock(in_channels, c_skip, kernel_size=
            kernel_size, dropout=dropout)
    elif block_type == 'SDCascadeAttnBlock':
        return SDCascadeAttnBlock(in_channels, conditioning_dim, nhead,
            self_attn=self_attn, dropout=dropout)
    elif block_type == 'SDCascadeTimestepBlock':
        return SDCascadeTimestepBlock(in_channels,
            timestep_ratio_embedding_dim, conds=timestep_conditioning_type)
    else:
        raise ValueError(f'Block type {block_type} not supported')
