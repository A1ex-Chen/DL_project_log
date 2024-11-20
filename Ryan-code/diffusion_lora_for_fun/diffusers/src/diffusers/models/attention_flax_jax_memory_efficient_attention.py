def jax_memory_efficient_attention(query, key, value, precision=jax.lax.
    Precision.HIGHEST, query_chunk_size: int=1024, key_chunk_size: int=4096):
    """
    Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
    https://github.com/AminRezaei0x443/memory-efficient-attention

    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
            numerical precision for computation
        query_chunk_size (`int`, *optional*, defaults to 1024):
            chunk size to divide query array value must divide query_length equally without remainder
        key_chunk_size (`int`, *optional*, defaults to 4096):
            chunk size to divide key and value array value must divide key_value_length equally without remainder

    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    num_q, num_heads, q_features = query.shape[-3:]

    def chunk_scanner(chunk_idx, _):
        query_chunk = jax.lax.dynamic_slice(operand=query, start_indices=[0
            ] * (query.ndim - 3) + [chunk_idx, 0, 0], slice_sizes=list(
            query.shape[:-3]) + [min(query_chunk_size, num_q), num_heads,
            q_features])
        return chunk_idx + query_chunk_size, _query_chunk_attention(query=
            query_chunk, key=key, value=value, precision=precision,
            key_chunk_size=key_chunk_size)
    _, res = jax.lax.scan(f=chunk_scanner, init=0, xs=None, length=math.
        ceil(num_q / query_chunk_size))
    return jnp.concatenate(res, axis=-3)
