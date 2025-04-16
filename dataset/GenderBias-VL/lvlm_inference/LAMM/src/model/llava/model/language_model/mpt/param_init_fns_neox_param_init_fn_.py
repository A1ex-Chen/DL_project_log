def neox_param_init_fn_(module: nn.Module, n_layers: int, d_model: int,
    emb_init_std: Optional[float]=None, emb_init_uniform_lim: Optional[
    Union[Tuple[float, float], float]]=None, verbose: int=0, **kwargs):
    """From section 2.3.1 of GPT-NeoX-20B:

    An Open-Source AutoregressiveLanguage Model — Black et. al. (2022)
    see https://github.com/EleutherAI/gpt-neox/blob/9610391ab319403cef079b438edd016a2443af54/megatron/model/init_functions.py#L151
    and https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py
    """
    del kwargs
    residual_div = n_layers / math.sqrt(10)
    if verbose > 1:
        warnings.warn(f'setting init_div_is_residual to {residual_div}')
    small_param_init_fn_(module=module, d_model=d_model, n_layers=n_layers,
        init_div_is_residual=residual_div, emb_init_std=emb_init_std,
        emb_init_uniform_lim=emb_init_uniform_lim, verbose=verbose)
