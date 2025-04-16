def __init__(self, config: CLIPConfig, input_shape: Optional[Tuple]=None,
    seed: int=0, dtype: jnp.dtype=jnp.float32, _do_init: bool=True, **kwargs):
    if input_shape is None:
        input_shape = 1, 224, 224, 3
    module = self.module_class(config=config, dtype=dtype, **kwargs)
    super().__init__(config, module, input_shape=input_shape, seed=seed,
        dtype=dtype, _do_init=_do_init)
