def shape(states):
    """  projection """
    return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
