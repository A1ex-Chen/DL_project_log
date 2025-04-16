def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0):
    if block_type == 'C':
        return ResBlockStageB(c_hidden, c_skip, kernel_size=kernel_size,
            dropout=dropout)
    elif block_type == 'A':
        return AttnBlock(c_hidden, c_cond, nhead, self_attn=True, dropout=
            dropout)
    elif block_type == 'T':
        return TimestepBlock(c_hidden, c_r)
    else:
        raise ValueError(f'Block type {block_type} not supported')
