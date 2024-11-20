def _get_least_common_mult_chunk_len(config):
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == 'lsh':
        return config.lsh_attn_chunk_length
    elif len(attn_types_set) == 1 and attn_types[0] == 'local':
        return config.local_attn_chunk_length
    elif len(attn_types_set) == 2 and attn_types_set == set(['lsh', 'local']):
        return np.lcm(config.lsh_attn_chunk_length, config.
            local_attn_chunk_length)
    else:
        raise NotImplementedError(
            "Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {}. Select attn layer types from ['lsh', 'local'] only."
            .format(config.attn_layers))
