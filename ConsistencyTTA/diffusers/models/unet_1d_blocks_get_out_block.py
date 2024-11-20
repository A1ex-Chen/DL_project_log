def get_out_block(*, out_block_type, num_groups_out, embed_dim,
    out_channels, act_fn, fc_dim):
    if out_block_type == 'OutConv1DBlock':
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == 'ValueFunction':
        return OutValueFunctionBlock(fc_dim, embed_dim)
    return None
