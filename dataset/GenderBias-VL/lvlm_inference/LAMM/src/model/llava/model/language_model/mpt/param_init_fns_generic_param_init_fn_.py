def generic_param_init_fn_(module: nn.Module, init_fn_, n_layers: int,
    d_model: Optional[int]=None, init_div_is_residual: Union[int, float,
    str, bool]=True, emb_init_std: Optional[float]=None,
    emb_init_uniform_lim: Optional[Union[Tuple[float, float], float]]=None,
    verbose: int=0, **kwargs):
    del kwargs
    if verbose > 1:
        warnings.warn(
            f'If model has bias parameters they are initialized to 0.')
    init_div_is_residual = init_div_is_residual
    if init_div_is_residual is False:
        div_is_residual = 1.0
    elif init_div_is_residual is True:
        div_is_residual = math.sqrt(2 * n_layers)
    elif isinstance(init_div_is_residual, float) or isinstance(
        init_div_is_residual, int):
        div_is_residual = init_div_is_residual
    elif isinstance(init_div_is_residual, str
        ) and init_div_is_residual.isnumeric():
        div_is_residual = float(init_div_is_residual)
    else:
        div_is_residual = 1.0
        raise ValueError(
            f'Expected init_div_is_residual to be boolean or numeric, got {init_div_is_residual}'
            )
    if init_div_is_residual is not False:
        if verbose > 1:
            warnings.warn(
                f'Initializing _is_residual layers then dividing them by {div_is_residual:.3f}. '
                 +
                f'Set `init_div_is_residual: false` in init config to disable this.'
                )
    if isinstance(module, nn.Linear):
        if hasattr(module, '_fused'):
            fused_init_helper_(module, init_fn_)
        else:
            init_fn_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        if init_div_is_residual is not False and getattr(module,
            '_is_residual', False):
            with torch.no_grad():
                module.weight.div_(div_is_residual)
    elif isinstance(module, nn.Embedding):
        if emb_init_std is not None:
            std = emb_init_std
            if std == 0:
                warnings.warn(f'Embedding layer initialized to 0.')
            emb_init_fn_ = partial(torch.nn.init.normal_, mean=0.0, std=std)
            if verbose > 1:
                warnings.warn(
                    f'Embedding layer initialized using normal distribution with mean=0 and std={std!r}.'
                    )
        elif emb_init_uniform_lim is not None:
            lim = emb_init_uniform_lim
            if isinstance(lim, Sequence):
                if len(lim) > 2:
                    raise ValueError(
                        f'Uniform init requires a min and a max limit. User input: {lim}.'
                        )
                if lim[0] == lim[1]:
                    warnings.warn(f'Embedding layer initialized to {lim[0]}.')
            else:
                if lim == 0:
                    warnings.warn(f'Embedding layer initialized to 0.')
                lim = [-lim, lim]
            a, b = lim
            emb_init_fn_ = partial(torch.nn.init.uniform_, a=a, b=b)
            if verbose > 1:
                warnings.warn(
                    f'Embedding layer initialized using uniform distribution in range {lim}.'
                    )
        else:
            emb_init_fn_ = init_fn_
        emb_init_fn_(module.weight)
    elif isinstance(module, tuple(set(NORM_CLASS_REGISTRY.values()))):
        if verbose > 1:
            warnings.warn(
                f'Norm weights are set to 1. If norm layer has a bias it is initialized to 0.'
                )
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:
            assert module.in_proj_weight is not None
            assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
            assert d_model is not None
            _d = d_model
            splits = 0, _d, 2 * _d, 3 * _d
            for s, e in zip(splits[:-1], splits[1:]):
                init_fn_(module.in_proj_weight[s:e])
        else:
            assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
            assert module.in_proj_weight is None
            init_fn_(module.q_proj_weight)
            init_fn_(module.k_proj_weight)
            init_fn_(module.v_proj_weight)
        if module.in_proj_bias is not None:
            torch.nn.init.zeros_(module.in_proj_bias)
        if module.bias_k is not None:
            torch.nn.init.zeros_(module.bias_k)
        if module.bias_v is not None:
            torch.nn.init.zeros_(module.bias_v)
        init_fn_(module.out_proj.weight)
        if init_div_is_residual is not False and getattr(module.out_proj,
            '_is_residual', False):
            with torch.no_grad():
                module.out_proj.weight.div_(div_is_residual)
        if module.out_proj.bias is not None:
            torch.nn.init.zeros_(module.out_proj.bias)
    else:
        for _ in module.parameters(recurse=False):
            raise NotImplementedError(
                f'{module.__class__.__name__} parameters are not initialized by param_init_fn.'
                )
