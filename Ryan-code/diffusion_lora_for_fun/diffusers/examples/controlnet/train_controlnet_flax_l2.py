def l2(xs):
    return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(
        xs)]))
