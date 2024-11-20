def get_out_block(*, out_block_type: str, num_groups_out: int, embed_dim:
    int, out_channels: int, act_fn: str, fc_dim: int) ->Optional[OutBlockType]:
    if out_block_type == 'OutConv1DBlock':
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == 'ValueFunction':
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn)
    return None
