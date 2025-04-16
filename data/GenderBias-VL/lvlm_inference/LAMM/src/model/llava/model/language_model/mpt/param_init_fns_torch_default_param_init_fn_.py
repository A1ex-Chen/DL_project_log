def torch_default_param_init_fn_(module: nn.Module, verbose: int=0, **kwargs):
    del kwargs
    if verbose > 1:
        warnings.warn(
            f"Initializing network using module's reset_parameters attribute")
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
