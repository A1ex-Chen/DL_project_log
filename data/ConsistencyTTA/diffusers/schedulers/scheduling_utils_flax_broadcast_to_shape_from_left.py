def broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int]
    ) ->jnp.ndarray:
    assert len(shape) >= x.ndim
    return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim
        )), shape)
