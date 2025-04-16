def small_param_init_fn_(module: nn.Module, n_layers: int, d_model: int,
    init_div_is_residual: Union[int, float, str, bool]=True, emb_init_std:
    Optional[float]=None, emb_init_uniform_lim: Optional[Union[Tuple[float,
    float], float]]=None, verbose: int=0, **kwargs):
    del kwargs
    std = math.sqrt(2 / (5 * d_model))
    _normal_param_init_fn_(module=module, std=std, d_model=d_model,
        n_layers=n_layers, init_div_is_residual=init_div_is_residual,
        emb_init_std=emb_init_std, emb_init_uniform_lim=
        emb_init_uniform_lim, verbose=verbose)
