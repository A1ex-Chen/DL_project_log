def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.
    dtype, mask: Any=None) ->Any:
    """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

    def conditional_cast(param):
        if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype,
            jnp.floating):
            param = param.astype(dtype)
        return param
    if mask is None:
        return jax.tree_map(conditional_cast, params)
    flat_params = flatten_dict(params)
    flat_mask, _ = jax.tree_flatten(mask)
    for masked, key in zip(flat_mask, flat_params.keys()):
        if masked:
            param = flat_params[key]
            flat_params[key] = conditional_cast(param)
    return unflatten_dict(flat_params)
