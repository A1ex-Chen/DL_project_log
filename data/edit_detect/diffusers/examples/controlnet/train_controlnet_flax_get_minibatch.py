def get_minibatch(batch, grad_idx):
    return jax.tree_util.tree_map(lambda x: jax.lax.dynamic_index_in_dim(x,
        grad_idx, keepdims=False), batch)
