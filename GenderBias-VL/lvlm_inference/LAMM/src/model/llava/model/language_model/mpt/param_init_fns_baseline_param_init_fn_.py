def baseline_param_init_fn_(module: nn.Module, init_std: float, n_layers:
    int, d_model: Optional[int]=None, init_div_is_residual: Union[int,
    float, str, bool]=True, emb_init_std: Optional[float]=None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None,
    verbose: int=0, **kwargs):
    del kwargs
    if init_std is None:
        raise ValueError(
            "You must set model.init_config['init_std'] to a float value to use the default initialization scheme."
            )
    _normal_param_init_fn_(module=module, std=init_std, d_model=d_model,
        n_layers=n_layers, init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std, emb_init_uniform_lim=
        emb_init_uniform_lim, verbose=verbose)
