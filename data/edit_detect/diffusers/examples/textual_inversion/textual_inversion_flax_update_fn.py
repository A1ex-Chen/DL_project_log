def update_fn(updates, state, params=None):
    return jax.tree_util.tree_map(jnp.zeros_like, updates), ()
