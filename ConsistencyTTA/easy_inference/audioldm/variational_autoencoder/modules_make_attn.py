def make_attn(in_channels, attn_type='vanilla'):
    assert attn_type in ['vanilla', 'linear', 'none'
        ], f'attn_type {attn_type} unknown'
    if attn_type == 'vanilla':
        return AttnBlock(in_channels)
    elif attn_type == 'none':
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)
