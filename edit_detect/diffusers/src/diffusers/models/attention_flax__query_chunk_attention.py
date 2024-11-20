def _query_chunk_attention(query, key, value, precision, key_chunk_size:
    int=4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum('...qhd,...khd->...qhk', query, key,
            precision=precision)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum('...vhf,...qhv->...qhf', value, exp_weights,
            precision=precision)
        max_score = jnp.einsum('...qhk->...qh', max_score)
        return exp_values, exp_weights.sum(axis=-1), max_score

    def chunk_scanner(chunk_idx):
        key_chunk = jax.lax.dynamic_slice(operand=key, start_indices=[0] *
            (key.ndim - 3) + [chunk_idx, 0, 0], slice_sizes=list(key.shape[
            :-3]) + [key_chunk_size, num_heads, k_features])
        value_chunk = jax.lax.dynamic_slice(operand=value, start_indices=[0
            ] * (value.ndim - 3) + [chunk_idx, 0, 0], slice_sizes=list(
            value.shape[:-3]) + [key_chunk_size, num_heads, v_features])
        return summarize_chunk(query, key_chunk, value_chunk)
    chunk_values, chunk_weights, chunk_max = jax.lax.map(f=chunk_scanner,
        xs=jnp.arange(0, num_kv, key_chunk_size))
    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs
    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights
