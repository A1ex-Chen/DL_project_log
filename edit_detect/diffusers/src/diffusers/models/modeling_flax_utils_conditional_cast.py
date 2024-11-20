def conditional_cast(param):
    if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.
        floating):
        param = param.astype(dtype)
    return param
