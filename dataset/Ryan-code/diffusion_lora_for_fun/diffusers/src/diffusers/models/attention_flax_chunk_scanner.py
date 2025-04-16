def chunk_scanner(chunk_idx, _):
    query_chunk = jax.lax.dynamic_slice(operand=query, start_indices=[0] *
        (query.ndim - 3) + [chunk_idx, 0, 0], slice_sizes=list(query.shape[
        :-3]) + [min(query_chunk_size, num_q), num_heads, q_features])
    return chunk_idx + query_chunk_size, _query_chunk_attention(query=
        query_chunk, key=key, value=value, precision=precision,
        key_chunk_size=key_chunk_size)
