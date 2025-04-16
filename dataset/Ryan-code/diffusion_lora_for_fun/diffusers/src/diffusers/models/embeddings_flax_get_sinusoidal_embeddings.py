def get_sinusoidal_embeddings(timesteps: jnp.ndarray, embedding_dim: int,
    freq_shift: float=1, min_timescale: float=1, max_timescale: float=
    10000.0, flip_sin_to_cos: bool=False, scale: float=1.0) ->jnp.ndarray:
    """Returns the positional encoding (same as Tensor2Tensor).

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        These may be fractional.
        embedding_dim: The number of output channels.
        min_timescale: The smallest time unit (should probably be 0.0).
        max_timescale: The largest time unit.
    Returns:
        a Tensor of timing signals [N, num_channels]
    """
    assert timesteps.ndim == 1, 'Timesteps should be a 1d-array'
    assert embedding_dim % 2 == 0, f'Embedding dimension {embedding_dim} should be even'
    num_timescales = float(embedding_dim // 2)
    log_timescale_increment = math.log(max_timescale / min_timescale) / (
        num_timescales - freq_shift)
    inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales,
        dtype=jnp.float32) * -log_timescale_increment)
    emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)
    scaled_time = scale * emb
    if flip_sin_to_cos:
        signal = jnp.concatenate([jnp.cos(scaled_time), jnp.sin(scaled_time
            )], axis=1)
    else:
        signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time
            )], axis=1)
    signal = jnp.reshape(signal, [jnp.shape(timesteps)[0], embedding_dim])
    return signal
