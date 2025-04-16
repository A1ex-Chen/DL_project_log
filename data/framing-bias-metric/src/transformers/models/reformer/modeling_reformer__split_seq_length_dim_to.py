def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2,
    num_attn_heads, attn_head_size=None):
    """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
    batch_size = vectors.shape[0]
    split_dim_shape = batch_size, num_attn_heads, dim_factor_1, dim_factor_2
    if len(vectors.shape) == 4:
        return torch.reshape(vectors, split_dim_shape + (attn_head_size,))
    elif len(vectors.shape) == 3:
        return torch.reshape(vectors, split_dim_shape)
    else:
        raise ValueError(
            'Input vector rank should be one of [3, 4], but is: {}'.format(
            len(vectors.shape)))
