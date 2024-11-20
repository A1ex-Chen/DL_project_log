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
