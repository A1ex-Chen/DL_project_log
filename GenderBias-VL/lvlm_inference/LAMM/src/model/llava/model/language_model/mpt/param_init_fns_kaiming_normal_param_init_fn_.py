def kaiming_normal_param_init_fn_(module: nn.Module, n_layers: int, d_model:
    Optional[int]=None, init_div_is_residual: Union[int, float, str, bool]=
    True, emb_init_std: Optional[float]=None, emb_init_uniform_lim:
    Optional[Union[Tuple[float, float], float]]=None, init_gain: float=0,
    fan_mode: str='fan_in', init_nonlinearity: str='leaky_relu', verbose:
    int=0, **kwargs):
    del kwargs
    if verbose > 1:
        warnings.warn(
            f'Using nn.init.kaiming_normal_ init fn with parameters: ' +
            f'a={init_gain}, mode={fan_mode}, nonlinearity={init_nonlinearity}'
            )
    kaiming_normal_ = partial(torch.nn.init.kaiming_normal_, a=init_gain,
        mode=fan_mode, nonlinearity=init_nonlinearity)
    generic_param_init_fn_(module=module, init_fn_=kaiming_normal_, d_model
        =d_model, n_layers=n_layers, init_div_is_residual=
        init_div_is_residual, emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim, verbose=verbose)
