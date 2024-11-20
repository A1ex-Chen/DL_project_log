def _normal_param_init_fn_(module: nn.Module, std: float, n_layers: int,
    d_model: Optional[int]=None, init_div_is_residual: Union[int, float,
    str, bool]=True, emb_init_std: Optional[float]=None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None,
    verbose: int=0, **kwargs):
    del kwargs
    init_fn_ = _normal_init_(std=std)
    if verbose > 1:
        warnings.warn(
            f'Using torch.nn.init.normal_ init fn mean=0.0, std={std}')
    generic_param_init_fn_(module=module, init_fn_=init_fn_, d_model=
        d_model, n_layers=n_layers, init_div_is_residual=
        init_div_is_residual, emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim, verbose=verbose)
